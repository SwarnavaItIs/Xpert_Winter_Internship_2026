"""Detection-only development metrics, faithfully using official greedy anchors."""
import re


def match_sources(candidates,labels):
    pairs=[]
    used=set()
    for i,candidate in enumerate(candidates):
        for j,label in enumerate(labels):
            if j not in used and candidate.get('source_msg') in label['source_msgs']:
                pairs.append((i,j))
                used.add(j)
                break
    return pairs


def scoped_labels(answers,development_message_ids,validation_message_ids):
    labels=[]
    boundary=0
    unresolved=0
    # Do not retain statuses, closure evidence, help or validation task descriptions.
    for row in answers.get('open_loops',[]):
        sources=set(row.get('source_msgs',[]))
        if sources & development_message_ids:
            if sources & validation_message_ids:
                boundary+=1
            elif not sources <= development_message_ids:
                unresolved+=1
            else:
                labels.append({k:row.get(k) for k in ('title','type','owner','chat_id','source_msgs')})
    return labels,{'cross_partition_source_tasks':boundary,'unknown_source_tasks':unresolved}


def detection_metrics(candidates,labels):
    pairs=match_sources(candidates,labels)
    tp=len(pairs)
    precision=tp/len(candidates) if candidates else 0.0
    recall=tp/len(labels) if labels else None
    f1=2*tp/(len(candidates)+len(labels)) if candidates and labels else 0.0
    return {'predicted':len(candidates),'labelled':len(labels),'matched':tp,
            'false_positives':len(candidates)-tp,'false_negatives':len(labels)-tp,
            'precision':precision,'recall':recall,'f1':f1},pairs


def anchor_diagnostics(candidates,labels,pairs,index):
    matched_p={i for i,_ in pairs}
    matched_g={j for _,j in pairs}
    possibilities=[]
    for i,c in enumerate(candidates):
        if i in matched_p:
            continue
        source=index.message[c['source_msg']]
        cwords=set(re.findall(r'\w+',c['title'].casefold()))
        for j,g in enumerate(labels):
            if j in matched_g or c['owner']!=g.get('owner'):
                continue
            gwords=set(re.findall(r'\w+',str(g.get('title','')).casefold()))
            overlap=len(cwords & gwords)/len(cwords | gwords) if cwords|gwords else 0
            anchors=[index.message[mid] for mid in g['source_msgs'] if mid in index.message]
            near=any(m['chat_id']==source['chat_id'] and abs(m['ts']-source['ts'])<=2*86400 for m in anchors)
            linked=source.get('quoted_id') in g['source_msgs'] or any(m.get('quoted_id')==source['id'] for m in anchors)
            exact=cwords==gwords and bool(cwords)
            if near and overlap>=.5 and (linked or exact):
                possibilities.append((overlap,i,j,'quote link and title overlap' if linked else 'same title/owner nearby'))
    suggestions=[]
    used_p,used_g=set(),set()
    for similarity,i,j,reason in sorted(possibilities,reverse=True):
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            suggestions.append({'candidate_id':candidates[i]['candidate_id'],
                'predicted_source':candidates[i]['source_msg'],'accepted_sources':labels[j]['source_msgs'],
                'label_index':j,'reason':reason,'title_token_jaccard':similarity,
                'review_status':'unreviewed; not credited as a match'})
    missing=[{'label_index':j,**g} for j,g in enumerate(labels) if j not in matched_g and j not in used_g]
    return {'suspected_anchor_errors':suggestions,'unmatched_without_anchor_suggestion':missing,
            'confirmed_anchor_errors':None,
            'note':'Automatic suggestions only. Remaining misses may also be anchor mistakes; semantic review is pending.'}
