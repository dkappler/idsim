

def register(obj_type, module):
    if issubclass(obj_type, module.Interface):
        module.FACTORY[obj_type.__name__] = obj_type
    else:
        import ipdb
        ipdb.set_trace()
        raise Exception(
            'The system type {} is not supported.'.format(obj_type))
