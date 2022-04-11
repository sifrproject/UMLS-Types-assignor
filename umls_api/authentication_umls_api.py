#!/usr/bin/python

# From https://github.com/HHS/uts-rest-api/blob/master/samples/python/Authentication.py
# See https://documentation.uts.nlm.nih.gov/rest/authentication.html for full explanation

import requests
import lxml.html as lh
from lxml.html import fromstring

uri = "https://utslogin.nlm.nih.gov"
auth_endpoint = "/cas/v1/api-key"

class Authentication:

    def __init__(self, apikey):
        self.apikey = apikey
        self.tgt = None
        self.service = "http://umlsks.nlm.nih.gov"
        self.gettgt()

    def gettgt(self):
        params = {'apikey': self.apikey}
        h = {"Content-type": "application/x-www-form-urlencoded",
             "Accept": "text/plain", "User-Agent": "python"}
        r = requests.post(uri+auth_endpoint, data=params, headers=h)
        response = fromstring(r.text)
        tgt = response.xpath('//form/@action')[0]
        tgt = tgt.replace("api-key", "tickets")
        self.tgt = tgt

    def getst(self):
        if self.tgt is None:
          self.gettgt()              
        params = {'service': self.service}
        h = {"Content-type": "application/x-www-form-urlencoded",
              "Accept": "text/plain", "User-Agent": "python"}
        r = requests.post(self.tgt, data=params, headers=h)
        st = r.text
        self.tgt = None
        return st
