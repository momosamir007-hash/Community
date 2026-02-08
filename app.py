import os
from cerebras.cloud.sdk import Cerebras

# ضع مفتاحك هنا
api_key = "مفتاحك_هنا"

client = Cerebras(api_key=api_key)

try:
    print("جاري البحث عن الموديلات المتاحة...")
    models = client.models.list()
    
    print("\n✅ الموديلات المتاحة لك هي:")
    for m in models:
        # طباعة اسم الموديل بالضبط لتنسخه
        print(f"- {m.id}") 

except Exception as e:
    print(f"حدث خطأ: {e}")
