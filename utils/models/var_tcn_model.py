from .tcn_model import *


class VarTcnV1(TcnV1):
    def forward(self, x, cont, cat):
        y = super().forward(x, cont, cat)
        y = 2 * nn.functional.sigmoid(2 * y)
        return y


class VarTcnV2(TcnV2):
    def forward(self, x, cont, cat):
        y = super().forward(x, cont, cat)
        y = 2 * nn.functional.sigmoid(2 * y)
        return y


class VarTcnV3(TcnV3):
    def forward(self, x, cont, cat):
        y = super().forward(x, cont, cat)
        y = 2 * nn.functional.sigmoid(2 * y)
        return y


class VarTcnV4(TcnV4):
    def forward(self, x, cont, cat):
        y = super().forward(x, cont, cat)
        y = 2 * nn.functional.sigmoid(2 * y)
        return y


class VarTcnV5(TcnV5):
    def forward(self, x, cont, cat):
        y = super().forward(x, cont, cat)
        y = 2 * nn.functional.sigmoid(2 * y)
        return y


var_tcn_models = dict(
    VarTcnV1=VarTcnV1,
    VarTcnV2=VarTcnV2,
    VarTcnV3=VarTcnV3,
    VarTcnV4=VarTcnV4,
    VarTcnV5=VarTcnV5,
)
