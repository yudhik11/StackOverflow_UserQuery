import requests
import numpy as np
u='https://api.stackexchange.com/2.2/tags'
tag_arr=[]
for i in range(1,555):
    print(i)
    p={'page':str(i), 'pagesize':'100', 'order':'desc','sort':'popular','site':'stackoverflow','key':'hWdB8OaWM0hGZP3sRV18iA(('}
    r = requests.get(url = u, params = p)
    data = r.json()
    temp = data['items']
    for i in range(0,len(temp)):
        tag_arr.append(data['items'][i]['name'])

np.savetxt('tags-so.txt', tag_arr, delimiter=',', newline='\n', fmt='%s')
