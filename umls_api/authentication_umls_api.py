#!/usr/bin/python

# From https://github.com/HHS/uts-rest-api/blob/master/samples/python/Authentication.py
# See https://documentation.uts.nlm.nih.gov/rest/authentication.html for full explanation

import requests
from lxml.html import fromstring

URI = "https://utslogin.nlm.nih.gov"
AUTH_ENDPOINT = "/cas/v1/api-key"


class Authentication:
    """Authentication for UMLS REST API"""

    def __init__(self, apikey: str):
        """Constructor

        Args:
            apikey (str): API key
        """
        self.apikey = apikey
        self.tgt = None
        self.service = "http://umlsks.nlm.nih.gov"
        self.gettgt()

    def gettgt(self):
        """Get the initial ticket granting ticket for future authentication"""
        params = {'apikey': self.apikey}
        headers = {"Content-type": "application/x-www-form-urlencoded",
                   "Accept": "text/plain", "User-Agent": "python"}
        response_request = requests.post(
            URI + AUTH_ENDPOINT, data=params, headers=headers)
        response = fromstring(response_request.text)
        try:
            tgt = response.xpath('//form/@action')[0]
            tgt = tgt.replace("api-key", "tickets")
            self.tgt = tgt
        except IndexError:
            print(response)
            print("Error getting TGT")
            self.tgt = "Error"

    def getst(self) -> str:
        """Get the service ticket for future authentication

        Returns:
            str: Service ticket
        """
        if self.tgt is None:
            self.gettgt()
        if self.tgt == "Error":
            return None
        params = {'service': self.service}
        headers = {"Content-type": "application/x-www-form-urlencoded",
                   "Accept": "text/plain", "User-Agent": "python"}
        response_request = requests.post(
            self.tgt, data=params, headers=headers)
        service_ticket = response_request.text
        self.tgt = None
        return service_ticket
