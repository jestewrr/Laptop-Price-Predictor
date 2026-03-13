import pickle, os
p='dual_price_models.pkl'
print('exists', os.path.exists(p))
if not os.path.exists(p):
    raise SystemExit('pickle not found')
with open(p,'rb') as f:
    data=pickle.load(f)
print('keys:', list(data.keys()))
for k in ('human_model','ai_model'):
    m=data.get(k)
    if m is None:
        print(k, '-> None')
        continue
    print(k, 'type:', type(m))
    print('module.name:', type(m).__module__ + '.' + type(m).__name__)
    # detect final estimator if pipeline
    if hasattr(m, 'named_steps'):
        print('pipeline steps:', list(m.named_steps.keys()))
        last = list(m.named_steps.values())[-1]
        print('pipeline final estimator type:', type(last).__module__ + '.' + type(last).__name__)
    # try sklearn estimator get_params
    try:
        params = getattr(m, 'get_params', None)
        if params:
            print('has get_params, sample keys:', list(m.get_params().keys())[:5])
    except Exception as e:
        print('get_params failed', e)
