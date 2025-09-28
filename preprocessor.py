import re
import pandas as pd

def preprocess(data):
    # UPDATED PATTERN: Now handles both '/' and '-' as date separators and both time formats
    pattern = re.compile(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}, \d{1,2}:\d{2}(?:\s?[ap]m)? - ', re.IGNORECASE)

    lines = data.split('\n')
    data_list = []
    
    for line in lines:
        if pattern.match(line):
            parts = line.split(' - ', 1)
            date_str = parts[0]
            
            user_msg_parts = parts[1].split(': ', 1)
            if len(user_msg_parts) > 1:
                user = user_msg_parts[0]
                message = user_msg_parts[1]
                data_list.append([date_str, user, message])
            else:
                user = 'group_notification'
                message = user_msg_parts[0]
                data_list.append([date_str, user, message])
        else:
            if data_list:
                data_list[-1][2] += "\n" + line

    df = pd.DataFrame(data_list, columns=['date_str', 'user', 'message'])

    # UPDATED SECTION: Tries multiple common date formats, including the new one
    def parse_datetime(s):
        formats = [
            '%m/%d/%y, %I:%M %p',  # NEW: MM/DD/YY, 12-hour AM/PM (e.g., 10/19/24, 2:19 AM)
            '%d/%m/%Y, %I:%M %p',  # DD/MM/YYYY, 12-hour AM/PM
            '%d/%m/%y, %I:%M %p',  # DD/MM/YY, 12-hour AM/PM
            '%d/%m/%Y, %H:%M',      # DD/MM/YYYY, 24-hour
            '%d/%m/%y, %H:%M',      # DD/MM/YY, 24-hour
        ]
        for fmt in formats:
            try:
                return pd.to_datetime(s.strip(), format=fmt)
            except ValueError:
                continue
        return pd.NaT

    df['date'] = df['date_str'].apply(parse_datetime)
    df.dropna(subset=['date'], inplace=True)
    df.drop(columns=['date_str'], inplace=True)

    # --- (Rest of the code is the same) ---
    if df.empty:
        return df # Return empty dataframe if no messages were parsed

    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    
    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))
    df['period'] = period
            
    return df