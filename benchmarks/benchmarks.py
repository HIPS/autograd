# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
# http://asv.readthedocs.io/en/latest/writing_benchmarks.html
from autograd import grad
import autograd.numpy as np
# from autograd.core import forward_pass, make_vjp
# from autograd.convenience_wrappers import safe_type,as_scalar,cast_to_same_dtype
# import torch
# from torch.autograd import Variable

class RNNSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
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

        # Set up for time_rnn_backward
        # self.rnn_forward()

    # def rnn_forward(self):
    #     def scalar_fun(*args, **kwargs):
    #         return as_scalar(self.fn(*args, **kwargs))
    #     argnum = 0
    #     args = list(((self.W1,self.b1,self.Wout,self.bout),self.x,self.l,self.n))
    #     args[argnum] = safe_type(args[argnum])
    #     self.vjp, self.ans = make_vjp(scalar_fun, argnum)(*args)
    #     pass

    # def rnn_backward(self):
    #     self.vjp(cast_to_same_dtype(1.0,self.ans))
    #     pass

    def rnn_grad(self):
        self.grad_fn((self.W1,self.b1,self.Wout,self.bout),self.x,self.l,self.n)

    # def time_rnn_forward(self):
    #     self.rnn_forward()

    # def peakmem_rnn_forward(self):
    #     self.rnn_forward()

    # def time_rnn_backward(self):
    #     self.rnn_backward()

    # def peakmem_rnn_backward(self):
    #     self.rnn_backward()

    def time_rnn_grad(self):
        self.rnn_grad()

    def peakmem_rnn_grad(self):
        self.rnn_grad()


# class PyTorchTimeSuite:
#     def setup(self):
#         self.batch_size = 16
#         self.dtype = "float32"
#         self.D = 2**10
#         self.x = 0.01 * np.random.randn(self.batch_size,self.D).astype(self.dtype)
#         self.W1 = 0.01 * np.random.randn(self.D,self.D).astype(self.dtype)
#         self.b1 = 0.01 * np.random.randn(self.D).astype(self.dtype)
#         self.Wout = 0.01 * np.random.randn(self.D,1).astype(self.dtype)
#         self.bout = 0.01 * np.random.randn(1).astype(self.dtype)
#         self.l = (np.random.rand(self.batch_size,1) > 0.5).astype(self.dtype)
#         self.n = 50

#         self.tx = Variable(torch.from_numpy(self.x),requires_grad=False)
#         self.tW1 = Variable(torch.from_numpy(self.W1),requires_grad=True)
#         self.tb1 = Variable(torch.from_numpy(self.b1),requires_grad=True)
#         self.tWout = Variable(torch.from_numpy(self.Wout),requires_grad=True)
#         self.tbout = Variable(torch.from_numpy(self.bout),requires_grad=True)
#         self.tl = Variable(torch.from_numpy(self.l))


#         def torch_rnn(x,W,b,Wout,bout,label,n):
#             h1 = x
#             for i in range(n):
#                 h1 = torch.tanh(torch.mm(h1,W) + torch.unsqueeze(b, 0).expand(x.size(0), b.size(0)))
#             logit = torch.mm(h1,Wout) + bout.expand(h1.size()[0])
#             loss = -torch.sum(label * logit - (
#                     logit + torch.log(1 + torch.exp(-logit))))
#             return loss

#         def grad_torch_rnn(x,W,b,Wout,bout,label,n):
#             loss = torch_rnn(x, W, b, Wout, bout, label, n)
#             loss.backward()
#             return loss, [W.grad,b.grad,Wout.grad,bout.grad]

#         self.fn = torch_rnn
#         self.grad_fn = grad_torch_rnn

#     def time_rnn_forward(self):
#         self.fn(self.tx,self.tW1,self.tb1,self.tWout,self.tbout,self.tl,self.n)
#         pass

#     def time_rnn_grad(self)
#         self.grad_fn(self.tx,self.tW1,self.tb1,self.tWout,self.tbout,self.tl,self.n)
#         pass

#     def time_rnn_grad(self):
#         self.grad_fn((self.W1,self.b1,self.Wout,self.bout),self.x,self.l,self.n)