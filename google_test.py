from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import os.path

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/calendar']


def authenticate():
    creds = None

    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    # if os.path.exists('token.json'):
    #     creds = Credentials.from_authorized_user_file('token.json')

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                '/home/gianfranco/Desktop/uni/masters-thesis/client_secret_959352362410-2ru9tokk20ck292fak2l332vtjkvtgia.apps.googleusercontent.com.json',
                SCOPES,
                redirect_uri='http://localhost:8000')
            authorization_url, _ = flow.authorization_url(prompt='consent')

        # Save the credentials for the next run
        # with open('token.json', 'w') as token:
        #     token.write(creds.to_json())

    return creds


authenticate()
