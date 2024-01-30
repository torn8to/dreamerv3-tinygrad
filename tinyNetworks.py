from tinygrad.tensor import Tensor as tensor
from tinygrad.dtype import dtypes
from tinygrad.nn import Linear, Conv2d, LayerNorm
import numpy as np
import tools
from torch import distributions as torchd
#
#The networks for the
#

class GRUCell:
    def __init__(self, inp_size, size, norm=True, act=tensor.tanh, update_bias=-1):
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = []
        self.layers.add(Linear(inp_size + size, 3 * size, bias=False))
        if norm:
            self.layers.add(LayerNorm(3 * size, eps=1e-03))


    @property
    def state_size(self):
        return self._size

    def forward(self, inputs:tensor, state:tensor):
        parts = tensor.cat([inputs, state], -1).sequential(self.layers)
        reset, cand, update = tensor.split(parts, [self._size] * 3, -1)
        reset = tensor.sigmoid(reset)
        cand = self._act(reset * cand)
        update = tensor.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]

class RSSM:
    def __init__(self,
            stoch,
            deter,
            hidden,
            rec_depth,
            discrete=False,
            activation= tensor.silu,
            norm=True,
            mean_act ="none",
            std_act=tensor.softplus,
            min_std=0.1,
            unimix_ratio=0.01,
            initial="learned",
            num_actions=None,
            embed=None,
            device=None,
                 ):
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete

        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        inp_layers.append(Linear(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(activation())
        self._img_in_layers:list[Linear|LayerNorm|tensor] = inp_layers
        self._img_in_layers.apply(tools.weight_init)
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)

        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(activation())
        self._img_out_layers = tensor.sequential(img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        inp_dim = self._deter + self._embed
        obs_out_layers.append(Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(activation())
        self._obs_out_layers = tensor.sequential(obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        if self._discrete:
            self._imgs_stat_layer = Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0)) #todo replace all kaiming weight initilizations with a tinyied version
        else:
            self._imgs_stat_layer = Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        if self._initial == "learned":
            self.W =  tensor.zeros((1, self._deter), device=tensor.device(self._device))
            self.W.requires_grad(True)
    def initial(self,batch_size):
        deter = self.zeros(batch_size, self._deter).to(self._device)
        if self._discrete:
            state = dict(
                logit=tensor.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=tensor.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=tensor.zeros([batch_size, self._stoch]).to(self._device),
                std=tensor.zeros([batch_size, self._stoch]).to(self._device),
                stoch=tensor.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = tensor.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"]) #TODO replace dict with something else maybe
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first,state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior


    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
        stoch = stoch.reshape(shape)
        return tensor.cat(stoch, state["deter"], dim=-1)


    #TODO get dist
    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist


    def obs_step(self, prev_state, prev_action, embed, is_first,sample=None):
        # initialize all prev_state
        if prev_state == None or tensor.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = tensor.zeros((len(is_first), self._num_actions)).to(
                self._device
            )
        # overwrite the prev_state only where is_first=True
        elif tensor.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = tensor.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        prior = self.img_step(prev_state, prev_action)
        x = tensor.cat(prior["deter"], embed, dim=-1)
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        x = self._obs_out_layers(x)
        # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("obs", x)#todo
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior


    def img_step(self,prev_state, prev_action,sample=None):
        # (batch, stoch, discrete_num)
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action)
        x = tensor.cat(prev_stoch, prev_action, dim =  -1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = tensor.sequential(x,self._img_in_layers)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        x = deter.sequential(self._img_out_layers)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()


    def _suff_stats_layer(self,name, x):
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = tensor.sequential(x,self._imgs_stat_layer)
            elif name == "obs":
                x = tensor.sequenital(x,self._obs_stat_layer)
            else:
                raise NotImplementedError
            mean, std = tensor.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * tensor.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: tensor.softplus(std),
                "abs": lambda: tensor.abs(std + 1),
                "sigmoid": lambda: tensor.sigmoid(std),
                "sigmoid2": lambda: 2 * tensor.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        rep_loss = tensor.clip(rep_loss, min=free)
        dyn_loss = tensor.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class MLP:
    def __init__(self,
                 inp_dim,
                 layers,
                 shape,
                 units,
                 activation=tensor.silu,
                 norm=True,
                 dist="normal",
                 std=1.0,
                 min_std=.1,
                 max_std=1.0,
                 absmax=False,
                 temp=0.1,
                 unimix_ratio=0.01,
                 out_scale=1.0,
                 symlog_inputs=False,
                 device="cuda"):
        self._shape = (shape,) if isinstance(shape,int) else shape
        self._dist = dist
        self._device = device
        self._activation = activation
        self._norm = norm
        self._temp = temp
        self._std = tensor((std,),dtype=dtypes.float,device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._unimix_ratio = unimix_ratio
        self._out_scale = out_scale
        self._symlog_inputs:bool = symlog_inputs
        self.layers:list = []
        for x in range(layers):
            self.layers.append(Linear(inp_dim,units,bias=False))
            self.layers.append(LayerNorm(units,eps=1e-03))
            self.layers.append(self._activation())
            inp_dim = units if x==0 else inp_dim
        if isinstance(self._shape,dict):
            self.mean_layer = [ ]
            for name, shape in self._shape.items():
                self.mean_layer.append(Linear(inp_dim,np.prod(shape)))
            if self._std =="learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layers = []
                for name, shape in self._shape.items():
                    self.std_layers.append(Linear(inp_dim,np.prod(shape)))
        elif self._shape is not None:
            self.mean_layer = [Linear(inp_dim, np.prod(self._shape))]
            if self._std =="learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layers = [Linear(units,np.prod(self._shape))]


    def __call__(self,x):
        x =  tools.symlog(x) if self._symlog_inputs == True else x
        out = x.sequential(self.layers)
        if self._shape is None: return out
        if isinstance(self._shape,dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)



    def dist(self, dist,mean,std,shape):
        pass
    #TODO implement dist




class MultiEncoder:
    def __init__(self,
                 shapes,
                 mlp_keys,
                 cnn_keys,
                 act,
                 norm,
                 cnn_depth,
                 kernel_size,
                 minres,
                 mlp_layers,
                 mlp_units,
                 symlog_inputs):
        pass

    def __call__(self):
        pass


class MultiDecoder:
    def __init__(self, ):
        pass

    def __call__(self, x):
        pass

    def _make_image_dist(self, mean):
        pass

class ImageEncoderResnet:
    pass




class ImageDecoderResnet:
    def __init__(self,):
        pass

    def __call__(self,):
        pass


