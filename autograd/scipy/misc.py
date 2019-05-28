from __future__ import absolute_import

import scipy.misc as osp_misc
from ..scipy import special

if hasattr(osp_misc, 'logsumexp'):
    logsumexp = special.logsumexp
