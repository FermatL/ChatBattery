import requests
from .domain_agent import Domain_Agent

MP_api_key = 'v0oRagoyuL8xHNaWZ3pUsN6zXR19DMgD'



class Search_Agent:
    @staticmethod
    def ICSD_search(formula, ICSD_DB):
        for ICSD_formula in ICSD_DB:
            if Domain_Agent.range_match(formula, ICSD_formula):
                return True
        return False

    @staticmethod
    def MP_search(formula):
        from mp_api.client import MPRester
        try:
            with MPRester(MP_api_key) as mpr:
                # exact match
                docs = mpr.summary.search(formula=formula)
            return len(docs) >= 1
        except:
            return False
