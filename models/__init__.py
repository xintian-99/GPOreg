from models.gaussianSplatComplex import gaussianSplatComplex

def getModel(opt):

    model_name = opt['model']
    nkwargs = opt['nkwargs']
    model = None

    if 'gaussianSplatComplex' in model_name:
        model = gaussianSplatComplex(**nkwargs)
    else:
        raise ValueError("Model %s not recognized." % model_name)

    model = model.cuda()
    print("----->>>> Model %s is built ..." % model_name)

    return model