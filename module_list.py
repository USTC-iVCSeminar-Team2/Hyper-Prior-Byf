import torch


def compressor_list(a, h, rank):
    if a.model_name == 'HyperPrior':
        from models.hyper_prior_compressor import HyperPrior
        model = HyperPrior(a, h, rank, N=192, M=320)
        print('Successfully load model: {}'.format(a.model_name))
        return model
    else:
        raise Exception('Cannot find model: {}'.format(a.model_name))


def optimizer_list(model, h):
    if h.optim_name == 'AdamW':
        optim = torch.optim.AdamW(model.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
        return optim
