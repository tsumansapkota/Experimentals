# import torch
# import torch.nn as nn
# from .flows import Flow


# class DoubleReluLinearFlow(Flow):

#     def __init__(self, dim, bias=True, identity_init=True):
#         super().__init__()
#         self.dim = dim

#         _l = nn.Linear(2*dim, dim, bias)
#         if identity_init:
#             self.weight = nn.Parameter(torch.eye(dim).expand(2, dim, dim).reshape(2*dim, dim))
#         else:
#             self.weight = nn.Parameter(_l.weight.data.t().reshape(dim*2, dim))
#         if bias:
#             self.bias = nn.Parameter(_l.bias.data)
#         else:
#             self.bias = None
#         del _l
#         self._restrict_weight()

#     def _restrict_weight(self):
#         # self.weight.data[:self.dim, :] = torch.abs(self.weight.data[:self.dim, :])\
#         #                                     *torch.sign(self.weight.data[self.dim:, :])
#         self.weight.data = torch.abs(self.weight.data)        
#         pass

#     def _forward_yes_logDetJ(self, x):
#         self._restrict_weight()
#         ### selects 1, aka lower portion of matrix if negative.
#         # indx = (x<0).type(torch.long)*self.dim+torch.arange(0, x.shape[1])
#         # W = self.weight[indx]
#         # y = (x.unsqueeze(1) @ W).squeeze(1)
#         # if self.bias is not None:
#         #     y = y + self.bias
#         # return y, self._logdetJ(W)

#         mask = x<0
#         indx = (mask).type(torch.long)*self.dim+torch.arange(0, x.shape[1])
#         yp = x @ self.weight[:self.dim, :]
#         yn = x @ self.weight[self.dim:, :]
#         y = torch.where(mask, yn, yp)

#         if self.bias is not None:
#             y = y + self.bias
#         return y, self._logdetJ(self.weight[indx])

#     def _forward_no_logDetJ(self, x):
#         self._restrict_weight()

#         mask = x<0
#         yp = x @ self.weight[:self.dim, :]
#         yn = x @ self.weight[self.dim:, :]
#         y = torch.where(mask, yn, yp)
#         if self.bias is not None:
#             y = y + self.bias
#         return y

#     def _inverse_yes_logDetJ(self, z):
#         self._restrict_weight()

#         if self.bias is not None:
#             z = z - self.bias
#         ### upper portion of matrix, positive inverse indicates it activates.
#         xp = z @ self.weight[:self.dim, :].inverse()
#         xn = z @ self.weight[self.dim:, :].inverse()

#         mask = xp<0
#         indx = (mask).type(torch.long)*self.dim+torch.arange(0, xp.shape[1])
#         x = torch.where(mask, xn, xp)

#         return x, -self._logdetJ(self.weight[indx]).expand(z.shape[0])

#     def _inverse_no_logDetJ(self, z):
#         self._restrict_weight()

#         if self.bias is not None:
#             z = z - self.bias
#         ### upper portion of matrix, positive inverse indicates it activates.
#         xp = z @ self.weight[:self.dim, :].inverse()
#         xn = z @ self.weight[self.dim:, :].inverse()

#         mask = xp<0
#         x = torch.where(mask, xn, xp)
#         return x


#     def _logdetJ(self, W):
#         # return torch.log(torch.abs(torch.det(self.weight))) 
#         ### torch.logdet gives nan if negative determinant
#         # return W.det().abs().log()
#         return W.det().log()


