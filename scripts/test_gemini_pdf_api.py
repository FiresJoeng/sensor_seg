from google import genai
from google.genai import types
import pathlib
import httpx

client = genai.Client(api_key="AIzaSyChEpUg4XvdyjEWSFQn_l5kgamlyQPa3Wo")

# Retrieve and encode the PDF byte
filepath = pathlib.Path(r'C:\Users\kk\sensor\sensor_seg\data\input\中国石油工程建设有限公司西南分公司\BBPF-120-01-770-IN-DAT-007_A_集气装置改扩建温度变送器数据表_中.pdf')

prompt = "Summarize this document"
response = client.models.generate_content(
  model="gemini-2.5-flash",
  contents=[
      types.Part.from_bytes(
        data=filepath.read_bytes(),
        mime_type='application/pdf',
      ),
      prompt])
print(response.text)