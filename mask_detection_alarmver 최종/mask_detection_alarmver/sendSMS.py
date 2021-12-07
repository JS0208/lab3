import os
from twilio.rest import Client

# Find these values at https://twilio.com/user/account
# To set up environmental variables, see http://twil.io/secure
account_sid = 'AC61824c041f0fb67fa2a97f955cfc3801'
auth_token = '9ec38d8378980ec1cfb85e1e90874414'

client = Client(account_sid, auth_token)

client.api.account.messages.create(
    to="+821040059208",
    from_="+13605154927",
    body="Hello")