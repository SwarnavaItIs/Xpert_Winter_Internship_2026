"""Date-only resolution anchored to evidence timestamps, never wall-clock time."""
from datetime import datetime, timedelta, timezone, date
import re

DATA_TIMEZONE = timezone(timedelta(hours=5, minutes=30))
TIMEZONE_NOTE = 'Assumed Asia/Kolkata UTC+05:30; official input documents Unix seconds but no conversation timezone.'


def resolve_due(text, timestamp, evidence_id, future=True):
    local = datetime.fromtimestamp(timestamp, DATA_TIMEZONE).date()
    result = {'date': None, 'expression': text, 'evidence': [evidence_id] if text else [],
              'anchor_timestamp': timestamp, 'anchor_local_date': local.isoformat(),
              'timezone': 'Asia/Kolkata', 'timezone_assumed': True,
              'kind': 'unspecified', 'uncertain': False, 'reason': 'No supported date expression'}
    if not text:
        return result
    s = text.casefold().strip()
    if re.search(r'\b(?:or|ya)\b',s):
        result.update(kind='unresolved',uncertain=True,reason='Alternative due expressions require clarification')
        return result
    value, uncertain, reason = None, False, ''
    absolute = re.search(r'\b(\d{4}-\d{2}-\d{2})\b',s)
    if absolute:
        try:
            value = date.fromisoformat(absolute[1])
        except ValueError:
            result.update(kind='unresolved',uncertain=True,reason='Invalid explicit date')
            return result
        result['kind'] = 'explicit'
        reason = 'ISO date in evidence'
    elif re.search(r'\b(day after tomorrow|parso|parson)\b|পরশু',s):
        value = local+timedelta(days=2 if future else -2)
        uncertain = 'day after tomorrow' not in s
        reason = 'Relative two-day expression; romanised/Bengali past-future ambiguity uses obligation context'
    elif re.search(r'\btomorrow\b|আগামীকাল',s):
        value, reason = local+timedelta(days=1), 'Next local calendar day'
    elif re.search(r'\b(today|tonight|aaj|aj)\b|आज|আজ',s):
        value, reason = local, 'Same local calendar day; time of day is not inferred'
    elif re.search(r'\bkal\b|कल',s):
        value = local+timedelta(days=1 if future else -1)
        uncertain,reason = True,'Kal can mean yesterday/tomorrow; future obligation assumed'
    else:
        duration = re.search(r'\b(?:in|within|after)\s+(\d{1,3})\s+(day|days|week|weeks)\b',s)
        weekday = re.search(r'\b(?:(next|this|by|on)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',s)
        numeric = re.search(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b',s)
        if duration:
            days = int(duration[1])*(7 if duration[2].startswith('week') else 1)
            value,reason = local+timedelta(days=days),'Calendar duration from evidence date'
        elif weekday:
            days = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
            offset = (days.index(weekday[2])-local.weekday())%7
            if weekday[1]=='next' and offset==0:
                offset=7
            value=local+timedelta(days=offset)
            uncertain=True
            reason='Nearest upcoming weekday; next/this convention is ambiguous'
        elif numeric:
            try:
                value=date(int(numeric[3]),int(numeric[2]),int(numeric[1]))
            except ValueError:
                result.update(kind='unresolved',uncertain=True,reason='Invalid numeric date')
                return result
            result['kind']='explicit'
            uncertain=int(numeric[1])<=12 and int(numeric[2])<=12
            reason='Day/month/year assumption; numeric ordering may be ambiguous'
    if value:
        result.update(date=value.isoformat(), kind='explicit' if result['kind']=='explicit' else 'relative',
                      uncertain=uncertain,reason=reason)
    else:
        result.update(kind='unresolved',uncertain=True,reason='Date phrase not resolved; no date invented')
    return result


def find_due_phrase(text):
    pattern = (r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b|'
               r'\bday after tomorrow\b|\b(?:in|within|after)\s+\d{1,3}\s+(?:days?|weeks?)\b|'
               r'\b(?:(?:next|this|by|on)\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|'
               r'\b(?:tomorrow|today|tonight|kal|aaj|aj|parso|parson)\b|আগামীকাল|পরশু|आज|कल|আজ')
    matches = list(re.finditer(pattern,text,re.I))
    return matches[0].group() if matches else ''
