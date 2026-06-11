import ee

c = ee.ServiceAccountCredentials(
    'tes1-998@form-sembako-chain.iam.gserviceaccount.com',
    '/mnt/c/Users/LABTI/Deforest.id/config/form-sembako-sa-key.json'
)
ee.Initialize(credentials=c, project='form-sembako-chain')

tasks = ee.data.getTaskList()
hl = [t for t in tasks if t.get('description', '').startswith('hl_sample_')]

print("=== ACTIVE (not FAILED) ===")
active = [t for t in hl if t.get('state') != 'FAILED']
for t in active:
    print(f"  {t.get('state')}: {t.get('description')}")

print(f"\n=== SUMMARY ===")
from collections import Counter
states = Counter(t.get('state') for t in hl)
for s, n in states.most_common():
    print(f"  {s}: {n}")
