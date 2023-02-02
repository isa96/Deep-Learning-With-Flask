import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'review': 'movie plain terrible!!!! slow acting, slow get point wooden character there. best part show iron maiden sing video theater thats it. end worth watch wait it!! character movie put sleep almost. avoid it!!!'})

print(r.json())