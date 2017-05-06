from pylab import *
from autograd import grad
import autograd.numpy as np
import torch
from torch.autograd import Variable
from memory_profiler import memory_usage
from autograd.core import forward_pass, make_vjp
from autograd.convenience_wrappers import safe_type,as_scalar,cast_to_same_dtype

batch_size = 16

dtype = "float32"
D = 2**10
x = 0.01 * np.random.randn(batch_size,D).astype(dtype)
W1 = 0.01 * np.random.randn(D,D).astype(dtype)
b1 = 0.01 * np.random.randn(D).astype(dtype)
Wout = 0.01 * np.random.randn(D,1).astype(dtype)
bout = 0.01 * np.random.randn(1).astype(dtype)
l = (np.random.rand(batch_size,1) > 0.5).astype(dtype)
n = 50

# Autograd
def autograd_rnn(params, x, label, n):
    W, b, Wout, bout = params
    h1 = x
    for i in range(n):
        h1 = np.tanh(np.dot(h1, W) + b)
    logit = np.dot(h1, Wout) + bout
    loss = -np.sum(label * logit - (
            logit + np.log(1 + np.exp(-logit))))
    return loss
grad_autograd_rnn = grad(autograd_rnn)


def f():
    # forward_pass(autograd_rnn,args,{})
    def scalar_fun(*args, **kwargs):
        return as_scalar(autograd_rnn(*args, **kwargs))
    argnum = 0
    args = list(((W1,b1,Wout,bout),x,l,n))
    args[argnum] = safe_type(args[argnum])
    vjp, ans = make_vjp(scalar_fun, argnum)(*args)
    return vjp,ans
def b(vjp,ans):
    return vjp(cast_to_same_dtype(1.0,ans))
def bb():
    return grad_autograd_rnn((W1,b1,Wout,bout),x,l,n)

mem_usage_autograd_forward,(vjp,ans) = memory_usage(f,interval=0.01,retval=True)
mem_usage_autograd_forward = np.array(mem_usage_autograd_forward)
init_mem = mem_usage_autograd_forward[0]
mem_usage_autograd_forward -= init_mem
mem_usage_autograd = np.array(memory_usage((b,(vjp,ans),{}),interval=0.01))
mem_usage_autograd -= init_mem
mem_usage_autograd2 = np.array(memory_usage(bb,interval=0.01))
mem_usage_autograd2 -= mem_usage_autograd2[0]


# PyTorch
tx = Variable(torch.from_numpy(x),requires_grad=False)
tW1 = Variable(torch.from_numpy(W1),requires_grad=True)
tb1 = Variable(torch.from_numpy(b1),requires_grad=True)
tWout = Variable(torch.from_numpy(Wout),requires_grad=True)
tbout = Variable(torch.from_numpy(bout),requires_grad=True)
tl = Variable(torch.from_numpy(l))
def torch_rnn(x,W,b,Wout,bout,label,n):
    h1 = x
    for i in range(n):
        h1 = torch.tanh(torch.mm(h1,W) + torch.unsqueeze(b, 0).expand(x.size(0), b.size(0)))
    logit = torch.mm(h1,Wout) + bout.expand(h1.size()[0])
    loss = -torch.sum(label * logit - (
            logit + torch.log(1 + torch.exp(-logit))))
    return loss

def grad_torch_rnn(x,W,b,Wout,bout,label,n):
    loss = torch_rnn(x, W, b, Wout, bout, label, n)
    loss.backward()
    return loss, [W.grad,b.grad,Wout.grad,bout.grad]

def f():
    torch_rnn(tx,tW1,tb1,tWout,tbout,tl,n)
def b():
    grad_torch_rnn(tx,tW1,tb1,tWout,tbout,tl,n)

mem_usage_torch = np.array(memory_usage(b,interval=0.01))
mem_usage_torch -= mem_usage_torch[0]

clf()
# plot(mem_usage_autograd_forward, label="Autograd (Forwards Only)")
# xx = np.arange(len(mem_usage_autograd))+len(mem_usage_autograd_forward)-1
# plot(xx,mem_usage_autograd, label="Autograd (Forwards & Backwards Sep)")
plot(mem_usage_autograd2, label="Autograd (Forwards & Backwards, Grad)")
plot(mem_usage_torch, label="PyTorch (Forwards & Backwards)")
ylabel("Memory Usage(MB)")
xlabel("Time (sec)")
xticks(xticks()[0],xticks()[0]/100)
# yscale("log")
legend()
show()