import ee
c = ee.ServiceAccountCredentials(
    'tes1-998@form-sembako-chain.iam.gserviceaccount.com',
    '/mnt/c/Users/LABTI/Deforest.id/config/form-sembako-sa-key.json'
)
ee.Initialize(credentials=c, project='form-sembako-chain')
print('AUTH OK')
