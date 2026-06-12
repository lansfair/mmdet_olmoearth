from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class FindHook(Hook):
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if batch_idx > 0:
            return
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        for i, (name, parm) in enumerate(model.named_parameters()):
            # if parm.requires_grad and parm.grad is None:
            #     # import pdb
            #     # pdb.set_trace()
            print(i, name, tuple(parm.shape))