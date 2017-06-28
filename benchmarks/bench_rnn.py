# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
# http://asv.readthedocs.io/en/latest/writing_benchmarks.html
from autograd import grad
import autograd.numpy as np

class RNNSuite:
    """
    Checking speed on a vanilla RNN.
    """

    # NOTE: this is run each time we run a benchmark.
    # Might want to switch to setup_cache, which has to return an object which is loaded and unpacked in setup().
    def setup(self):
        self.batch_size = 16
        self.dtype = "float32"
        self.D = 2**10
        self.x = 0.01 * np.random.randn(self.batch_size,self.D).astype(self.dtype)
        self.W1 = 0.01 * np.random.randn(self.D,self.D).astype(self.dtype)
        self.b1 = 0.01 * np.random.randn(self.D).astype(self.dtype)
        self.Wout = 0.01 * np.random.randn(self.D,1).astype(self.dtype)
        self.bout = 0.01 * np.random.randn(1).astype(self.dtype)
        self.l = (np.random.rand(self.batch_size,1) > 0.5).astype(self.dtype)
        self.n = 50

        def autograd_rnn(params, x, label, n):
            W, b, Wout, bout = params
            h1 = x
            for i in range(n):
                h1 = np.tanh(np.dot(h1, W) + b)
            logit = np.dot(h1, Wout) + bout
            loss = -np.sum(label * logit - (
                    logit + np.log(1 + np.exp(-logit))))
            return loss

        self.fn = autograd_rnn
        self.grad_fn = grad(self.fn)

    def rnn_grad(self):
        self.grad_fn((self.W1,self.b1,self.Wout,self.bout),self.x,self.l,self.n)

    def time_rnn_grad(self):
        self.rnn_grad()

    def peakmem_rnn_grad(self):
        self.rnn_grad()

    def time_manual_rnn_grad(self):
        self.manual_rnn_grad()

    def peakmem_manual_rnn_grad(self):
        self.manual_rnn_grad()

    def manual_rnn_grad(self):
        def repeat_to_match_shape(g,A,axis=None):
            gout = np.empty_like(A)
            if np.ndim(gout) == 0:
                gout = g
            else:
                gout = np.ones_like(A)*g
            return gout

        def sum_to_match_shape(sum_this, to_match_this):
            sum_this = np.sum(sum_this,axis=tuple(range(0,np.ndim(sum_this)-np.ndim(to_match_this))))
            for axis, size in enumerate(np.shape(to_match_this)):
                if size == 1:
                    sum_this = np.sum(sum_this, axis=axis, keepdims=True)
            return sum_this

        def grad_dot_A(g,A,B):
            ga = np.dot(g,B.T)
            ga = np.reshape(ga,np.shape(A))
            return ga

        def grad_dot_B(g,A,B):
            gb = np.dot(A.T,g)
            gb = np.reshape(gb, np.shape(B))
            return gb

        def _rnn_grad(x, W, b, Wout, bout, label, n):
            h1__1_stack, h1__1 = [], None
            h1__0_stack, h1__0 = [], None
            out_stack, out = [], None
            h1_stack = []
            h1 = x
            _for1 = list(range(n))

            for i in _for1:
                h1__1_stack.append(h1__1)
                h1__1 = np.dot(h1, W)
                h1__0_stack.append(h1__0)
                h1__0 = h1__1 + b
                h1_stack.append(h1)
                h1 = np.tanh(h1__0)
            out__0 = np.dot(h1, Wout)
            out = out__0 + bout
            loss__2 = label * out
            loss__7 = -out
            loss__6 = np.exp(loss__7)
            loss__5 = 1 + loss__6
            loss__4 = np.log(loss__5)
            loss__3 = out + loss__4
            loss__1 = loss__2 - loss__3

            # Begin Backward Pass
            g_loss = 1
            g_h1__0 = 0
            g_h1__1 = 0
            g_b = 0
            g_W = 0

            # Reverse of: loss = -loss__0
            g_loss__0 = -g_loss

            # Reverse of: loss__0 = np.sum(loss__1)
            g_loss__1 = repeat_to_match_shape(g_loss__0, loss__1)

            # Reverse of: loss__1 = loss__2 - loss__3
            g_loss__2 = sum_to_match_shape(g_loss__1, loss__2)
            g_loss__3 = sum_to_match_shape(-g_loss__1, loss__3)

            # Reverse of: loss__3 = out + loss__4
            g_out = sum_to_match_shape(g_loss__3, out)
            g_loss__4 = sum_to_match_shape(g_loss__3, loss__4)

            # Reverse of: loss__4 = np.log(loss__5)
            g_loss__5 = g_loss__4 / loss__5

            # Reverse of: loss__5 = 1 + loss__6
            g_loss__6 = sum_to_match_shape(g_loss__5, loss__6)

            # Reverse of: loss__6 = np.exp(loss__7)
            g_loss__7 = g_loss__6 * np.exp(loss__7)

            # Reverse of: loss__7 = -out
            g_out += -g_loss__7
            g_out += sum_to_match_shape(g_loss__2 * label, out)

            # Reverse of: out = out__0 + bout
            g_out__0 = sum_to_match_shape(g_out, out__0)
            g_bout = sum_to_match_shape(g_out, bout)

            # Reverse of: out__0 = np.dot(h1, Wout)
            g_h1 = grad_dot_A(g_out__0, h1, Wout)
            g_Wout = grad_dot_B(g_out__0, h1, Wout)
            _for1 = reversed(_for1)
            for i in _for1:
                h1 = h1_stack.pop()
                tmp_g0 = g_h1 / np.cosh(h1__0) ** 2.0
                g_h1 = 0
                g_h1__0 += tmp_g0
                h1__0 = h1__0_stack.pop()
                tmp_g1 = sum_to_match_shape(g_h1__0, h1__1)
                tmp_g2 = sum_to_match_shape(g_h1__0, b)
                g_h1__0 = 0
                g_h1__1 += tmp_g1
                g_b += tmp_g2
                h1__1 = h1__1_stack.pop()
                tmp_g3 = grad_dot_A(g_h1__1, h1, W)
                tmp_g4 = grad_dot_B(g_h1__1, h1, W)
                g_h1__1 = 0
                g_h1 += tmp_g3
                g_W += tmp_g4
            return g_W, g_b, g_Wout, g_bout
        _rnn_grad(self.x, self.W1,self.b1,self.Wout,self.bout,self.l,self.n)
        pass
