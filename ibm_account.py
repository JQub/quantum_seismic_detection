from qiskit import IBMQ

#change the api for self
api_token = '22f33f85e8f8144e9fa2b2dffa0b3a2fccf8396564670db4b52c0d00250263e47799f278c9584ebdb98964a655e806e4ff05fe73bd913f3496d3345a0abe43bd'
hub  = 'ibm-q-education' #'ibm-q-education' 
group = 'george-mason-uni-1' #'george-mason-uni-1'
project ='hardware-acceler'#'hardware-acceler'
IBMQ.save_account(token = api_token,hub= hub,group =group,project = project,overwrite=True)
# IBMQ.save_account(token = api_token,overwrite=True)
# IBMQ.delete_account()
# IBMQ.active_account()
# IBMQ.save_account(token = api_token,overwrite=True)

provider = IBMQ.load_account()
print(provider)

# IBMQ.update_account()

# providers =IBMQ.providers()
# provider1 = IBMQ.get_provider(hub='ibm-q-education', group='george-mason-uni-1', project='hardware-acceler')
# print(provider1)