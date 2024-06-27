from abc import ABC, abstractmethod
from typing import Dict, Generic, Iterable, List, Optional, Type, TypeVar, cast
from typing_extensions import Self  # FUTURE: in typing from 3.11

from cirkit.new.layers import InputLayer, Layer, ProductLayer, SumLayer
from cirkit.new.utils import Scope
from cirkit.new.utils.type_aliases import SymbLayerCfg

LayerT_co = TypeVar("LayerT_co", bound=Layer, covariant=True)


# NOTE: This is generic corresponding to SymbLayerCfg, so that subclasses can refine internal types.
#       In most cases we use SymbolicLayer instead of GenericSymbolicLayer, so this class is not
#       included in __init__.py. Yet we still name it as public in case it need to be used.
class GenericSymbolicLayer(ABC, Generic[LayerT_co]):
    """The abstract base class for symbolic layers in symbolic circuits, with generics over the \
    concrete Layer class."""

    # We accept structure as positional args, and layer config as kw-only.
    # NOTE: The layers_in may contain any Layer, not just LayerT_co, because all layers can be used
    #       mixed in SymbC.
    def __init__(
        self,
        scope: Scope,
        layers_in: Iterable["SymbolicLayer"],
        *,
        num_units: int,
        layer_cfg: SymbLayerCfg[LayerT_co],
    ) -> None:
        """Construct the SymbolicLayer.

        Args:
            scope (Scope): The scope of this layer.
            layers_in (Iterable[SymbolicLayer]): The input to this layer, empty for input layers.
            num_units (int): The number of units in this layer.
            layer_cfg (SymbLayerCfg[LayerT_co]): The config for the concrete layer in symbolic form.
        """
        super().__init__()
        self.scope = scope

        # self.inputs is filled using layers_in, while self.outputs is empty until self appears in
        # another layer's layers_in. No need to de-duplicate, so prefer list over OrderedSet. Both
        # lists automatically gain a consistent ordering with RGNode edge tables by design.
        # ANNOTATE: Specify content for empty container.
        self.inputs: List["SymbolicLayer"] = []
        self.outputs: List["SymbolicLayer"] = []
        for layer_in in layers_in:
            self.inputs.append(layer_in)
            layer_in.outputs.append(self)

        self.arity = len(self.inputs)
        self.num_units = num_units
        # A SymbLayer instance should bind to a reparam instance but not just factory, to enable
        # reusing the same reparam in transforms.
        # IGNORE: Unavoidable for kwargs.
        self.layer_cfg = (
            SymbLayerCfg(  # TODO: better way to construct SymbLayerCfg than untyped replace?
                layer_cls=layer_cfg.layer_cls,
                layer_kwargs=layer_cfg.layer_kwargs,  # type: ignore[misc]
                reparam=layer_cfg.reparam_factory(),
                reparam_factory=None,
            )
            if layer_cfg.reparam_factory is not None
            else layer_cfg
        )
        self.layer_cls = layer_cfg.layer_cls  # Commonly used, so shorten reference.

    def __repr__(self) -> str:
        """Generate the repr string of the layer.

        Returns:
            str: The str representation of the layer.
        """
        # TODO: line folding of the repr string?
        repr_kv = ", ".join(f"{k}={v}" for k, v in self._repr_dict.items())
        return f"{type(self).__name__}@0x{id(self):x}({repr_kv})"

    # We use an abstract instead of direct attribute so that this class includes an abstract method.
    # NOTE: Use object to avoid Any. object is sufficient because we only need __repr__.
    @property
    @abstractmethod
    def _repr_dict(self) -> Dict[str, object]:
        """The dict of key-value pairs used in __repr__."""

    # __hash__ and __eq__ are defined by default to compare on object identity, i.e.,
    # (a is b) <=> (a == b) <=> (hash(a) == hash(b)).

    def transform(
        self,
        layers_in: Iterable["SymbolicLayer"],
        *,
        num_units: Optional[int] = None,
        layer_cfg: Optional[SymbLayerCfg[LayerT_co]] = None,
    ) -> Self:
        """Transform the SymbolicLayer into another SymbolicLayer with given input connections and \
        optionally some config overrides.

        This method is convenient for constructing a SymbLayer in another SymbC that corresponds \
        to this one.

        Args:
            layers_in (Iterable[SymbolicLayer]): The inputs to the new SymbolicLayer.
            num_units (Optional[int], optional): If provided, will override the default of \
                self.num_units. Defaults to None.
            layer_cfg (Optional[SymbLayerCfg[LayerT_co]], optional): If provided, will override \
                the default of self.layer_cfg. The layer_cls must be compatible to self.layer_cls. \
                Note that the reparam object will be shared by default. Defaults to None.

        Returns:
            Self: A new SymbolicLayer of the same type as self.
        """
        num_units = self.num_units if num_units is None else num_units
        # IGNORE: Unavoidable for kwargs.
        layer_cfg = (
            self.layer_cfg
            if layer_cfg is None
            else layer_cfg
            if layer_cfg.reparam_factory is None
            else SymbLayerCfg(  # TODO: better way to construct SymbLayerCfg than untyped replace?
                layer_cls=layer_cfg.layer_cls,
                layer_kwargs=layer_cfg.layer_kwargs,  # type: ignore[misc]
                reparam=layer_cfg.reparam_factory(),
                reparam_factory=None,
            )
        )
        return type(self)(self.scope, layers_in, num_units=num_units, layer_cfg=layer_cfg)

    def concretize(
        self,
        *,
        num_input_units: Optional[int] = None,
        num_output_units: Optional[int] = None,
        arity: Optional[int] = None,
    ) -> LayerT_co:
        """Concretize the SymbolicLayer into a Layer with the config saved and optinally some \
        config overrides.

        This method is convenient for constructing the Layer corresponding to this SymbolicLayer.

        Note that multiple calls to this methods will construct different Layer objects, but the \
        reference to internal reparam objects will always be the same.

        Args:
            num_input_units (Optional[int], optional): If provided, will override the default of \
                self.inputs[0].num_units. Must be provided if inputs is empty. Defaults to None.
            num_output_units (Optional[int], optional): If provided, will override the default of \
                self.num_units. Defaults to None.
            arity (Optional[int], optional): If provided, will override the default of self.arity. \
                Defaults to None.

        Returns:
            LayerT_co: The concrete Layer.
        """
        assert (
            self.layer_cfg.reparam_factory is None
        ), "A SymbL must hold an instance of reparam instead of a factory."
        num_input_units = self.inputs[0].num_units if num_input_units is None else num_input_units
        num_output_units = self.num_units if num_output_units is None else num_output_units
        arity = self.arity if arity is None else arity
        # CAST: We use a pair of redundant casts to enable intellisense pointing to Layer.__init__.
        # IGNORE: Unavoidable for kwargs.
        return cast(
            LayerT_co,
            cast(Type[Layer], self.layer_cls)(
                num_input_units=num_input_units,
                num_output_units=num_output_units,
                arity=arity,
                reparam=self.layer_cfg.reparam,
                **self.layer_cfg.layer_kwargs,  # type: ignore[misc]
            ),
        )


# This is a specialized version of the generic one, serving as the base of all possible SymbL.
SymbolicLayer = GenericSymbolicLayer[Layer]
"""The abstract base class for symbolic layers in symbolic circuits."""


class SymbolicSumLayer(GenericSymbolicLayer[SumLayer]):
    """The sum layer in symbolic circuits."""

    def __init__(
        self,
        scope: Scope,
        layers_in: Iterable[SymbolicLayer],
        *,
        num_units: int,
        layer_cfg: SymbLayerCfg[SumLayer],
    ) -> None:
        """Construct the SymbolicSumLayer.

        Args:
            scope (Scope): The scope of this layer.
            layers_in (Iterable[SymbolicLayer]): The input to this layer.
            num_units (int): The number of units in this layer.
            layer_cfg (SymbLayerCfg[SumLayer]): The config for the concrete layer in symbolic form.
        """
        super().__init__(scope, layers_in, num_units=num_units, layer_cfg=layer_cfg)
        # NOTE: layers_in may not support len(), so must check after __init__. Same for the
        #       following two classes.
        assert self.inputs, "SymbolicSumLayer must be an inner layer of the SymbC."

    @property
    def _repr_dict(self) -> Dict[str, object]:
        """The dict of key-value pairs used in __repr__."""
        return {
            "scope": self.scope,
            "arity": self.arity,
            "num_units": self.num_units,
            "layer_cfg": self.layer_cfg,
        }


class SymbolicProductLayer(GenericSymbolicLayer[ProductLayer]):
    """The product layer in symbolic circuits."""

    def __init__(
        self,
        scope: Scope,
        layers_in: Iterable[SymbolicLayer],
        *,
        num_units: int,
        layer_cfg: SymbLayerCfg[ProductLayer],
    ) -> None:
        """Construct the SymbolicProductLayer.

        Args:
            scope (Scope): The scope of this layer.
            layers_in (Iterable[SymbolicLayer]): The input to this layer.
            num_units (int): The number of units in this layer.
            layer_cfg (SymbLayerCfg[ProductLayer]): The config for the concrete layer in symbolic \
                form.
        """
        super().__init__(scope, layers_in, num_units=num_units, layer_cfg=layer_cfg)
        assert self.inputs, "SymbolicProductLayer must be an inner layer of the SymbC."

    @property
    def _repr_dict(self) -> Dict[str, object]:
        """The dict of key-value pairs used in __repr__."""
        return {
            "scope": self.scope,
            "arity": self.arity,
            "num_units": self.num_units,
            "layer_cfg": self.layer_cfg,
        }


class SymbolicInputLayer(GenericSymbolicLayer[InputLayer]):
    """The input layer in symbolic circuits."""

    def __init__(
        self,
        scope: Scope,
        layers_in: Iterable[SymbolicLayer],
        *,
        num_units: int,
        layer_cfg: SymbLayerCfg[InputLayer],
    ) -> None:
        """Construct the SymbolicInputLayer.

        Args:
            scope (Scope): The scope of this layer.
            layers_in (Iterable[SymbolicLayer]): The input to this layer, empty.
            num_units (int): The number of units in this layer.
            layer_cfg (SymbLayerCfg[InputLayer]): The config for the concrete layer in symbolic \
                form.
        """
        super().__init__(scope, layers_in, num_units=num_units, layer_cfg=layer_cfg)
        assert not self.inputs, "SymbolicInputLayer must be an input layer of the SymbC."

    @property
    def _repr_dict(self) -> Dict[str, object]:
        """The dict of key-value pairs used in __repr__."""
        return {
            "scope": self.scope,
            "num_units": self.num_units,
            "layer_cfg": self.layer_cfg,
        }
