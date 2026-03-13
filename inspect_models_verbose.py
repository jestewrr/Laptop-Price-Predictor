import pickle, os, traceback
p='dual_price_models.pkl'
print('exists', os.path.exists(p))
try:
    with open(p,'rb') as f:
        data=pickle.load(f)
except Exception as e:
    print('pickle load failed')
    traceback.print_exc()
    raise SystemExit(1)
print('loaded, keys:', list(data.keys()))
for k,v in data.items():
    try:
        print('key:', k, 'type:', type(v).__module__ + '.' + type(v).__name__)
    except Exception as e:
        print('key:', k, 'type-print failed', e)
    if hasattr(v, 'named_steps'):
        print('  pipeline steps:', list(v.named_steps.keys()))
        last = list(v.named_steps.values())[-1]
        print('  final estimator:', type(last).__module__ + '.' + type(last).__name__)
    
    try:
        if hasattr(v, 'get_params'):
            print('  get_params keys sample:', list(v.get_params().keys())[:5])
    except Exception as e:
        print('  get_params failed', e)
