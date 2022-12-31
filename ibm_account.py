from qiskit import IBMQ

api_token = 'your own token'
hub  = 'xxx' 
group = 'xxx' 
project ='xxx'
IBMQ.save_account(token = api_token,hub= hub,group =group,project = project,overwrite=True)
provider = IBMQ.load_account()
print(provider)