import scipy.misc as osp_misc
from ..scipy import special

if hasattr(osp_misc, "logsumexp"):
    logsumexp = special.logsumexp
