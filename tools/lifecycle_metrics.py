"""Explicit lifecycle denominators; no semantic duplicate claims from source-only labels."""
from tools.task_metrics import match_sources, detection_metrics


def project_lifecycle_labels(answers,development_ids,validation_ids):
    labels=[]
    excluded={'cross_partition_source_tasks':0,'unknown_source_tasks':0}
    for row in answers.get('open_loops',[]):
        sources=set(row.get('source_msgs',[]))
        if not sources & development_ids:
            continue
        if sources & validation_ids:
            excluded['cross_partition_source_tasks']+=1
        elif not sources <= development_ids:
            excluded['unknown_source_tasks']+=1
        else:
            labels.append({k:row.get(k) for k in ('source_msgs','status','closing_msgs')})
    return labels,excluded


def lifecycle_metrics(loops,gold,visible_message_ids):
    detection,pairs=detection_metrics(loops,gold)
    paired_gold={j for _,j in pairs}
    paired_pred={i for i,_ in pairs}
    counts={'matched':len(pairs),'matched_gold_open':0,'matched_predicted_closed':0,
            'false_closures':0,'missed_closures_matched':0,'missed_closures_with_visible_evidence':0,
            'wrong_closed_status':0,'correct_status':0,'closure_anchor_mismatches':0,
            'matched_closed_labels_without_visible_evidence':0,
            'matched_closed_labels_without_annotated_evidence':0,
            'all_predicted_closed':sum(p['status'] in ('done','dropped') for p in loops),
            'unmatched_predicted_closed':sum(p['status'] in ('done','dropped') for i,p in enumerate(loops) if i not in paired_pred)}
    for i,j in pairs:
        predicted,actual=loops[i],gold[j]
        pclosed=predicted['status'] in ('done','dropped')
        aclosed=actual['status'] in ('done','dropped')
        visible=bool(set(actual.get('closing_msgs') or []) & visible_message_ids)
        counts['matched_gold_open']+=actual['status']=='open'
        counts['matched_predicted_closed']+=pclosed
        counts['correct_status']+=predicted['status']==actual['status']
        counts['false_closures']+=pclosed and actual['status']=='open'
        counts['missed_closures_matched']+=not pclosed and aclosed
        counts['missed_closures_with_visible_evidence']+=not pclosed and aclosed and visible
        counts['matched_closed_labels_without_visible_evidence']+=aclosed and not visible
        counts['matched_closed_labels_without_annotated_evidence']+=aclosed and not actual.get('closing_msgs')
        counts['wrong_closed_status']+=pclosed and aclosed and predicted['status']!=actual['status']
        counts['closure_anchor_mismatches']+=pclosed and aclosed and bool(actual.get('closing_msgs')) and predicted.get('closed_by') not in actual['closing_msgs']
    counts['closed_tasks_missed_at_extraction']=sum(g['status'] in ('done','dropped') for j,g in enumerate(gold) if j not in paired_gold)
    counts['false_closure_rate_on_matched_open']=counts['false_closures']/counts['matched_gold_open'] if counts['matched_gold_open'] else None
    counts['false_closure_fraction_of_matched_closed_predictions']=counts['false_closures']/counts['matched_predicted_closed'] if counts['matched_predicted_closed'] else None
    counts['duplicate_anchor_predictions']=sum(i not in paired_pred and any(loop['source_msg'] in g['source_msgs'] for g in gold) for i,loop in enumerate(loops))
    counts['duplicate_note']='Excess predictions accepting an already-used labelled source: proxy only, not proven semantic duplicates.'
    return {'detection':detection,'lifecycle':counts}
