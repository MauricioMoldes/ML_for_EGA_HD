#!/usr/bin/env python

""" ml_rt.py: o HelpDesk Request Tracker ml exploration"""

__author__ = "Mauricio Moldes"
__version__ = "0.1"
__maintainer__ = "Mauricio Moldes"
__email__ = "mauricio.moldes@crg.eu"
__status__ = "Developement"

import logging
import sys
import yaml
import psycopg2
import rt
import csv
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import numpy as np
from spacy.lang.en import English
import spacy


def convert(message, threshold=.9):
    pos_tagger = English()  # part-of-speech tagger
    sentences = corpus2sentences(message)  # convert to sentences
    # iterate through sentence, write to a new file if not signature block
    converted_mail_body = generate_text(sentences)
    return converted_mail_body


def read_email(fname):
    with open(fname, 'r') as email:
        text = email.read()
    return text


def corpus2sentences(corpus):
    """split corpus into a list of sentences.
    """
    split_sentences = corpus.strip().split('\n')
    split_sentences_clean = list(filter(None, split_sentences))
    return split_sentences_clean


def generate_text(sentences, threshold=0.9):
    """iterate through sentences. if the sentence is not a signature block,
    write to file.

    if probability(signature block) > threshold, then it is a signature block.

    Parameters
    ----------
    sentence : str
        Represents line in email block.
    POS_parser: obj
        Spacy English object used to tag parts-of-speech. Will explore using
        other POS taggers like NLTK's.
    fname : str
        Represents fname of new corpus, excluding signature block.
    threshold: float
        Lower thresholds will result in more false positives.
    """
    tagger = spacy.load('en_core_web_sm')
    message_body = []
    for sentence in sentences:
        if prob_block(sentence, tagger) < threshold:
            message_body.append(sentence)
    return message_body


def prob_block(sentence, pos_tagger):
    """Calculate probability that a sentence is an email block.

    https://spacy.io/usage/linguistic-features

    Parameters
    ----------
    sentence : str
        Line in email block.

    Returns
    -------
    probability(signature block | line)
    """

    doc = pos_tagger(sentence)
    verb_count = np.sum([token.pos_ != "VERB" for token in doc])
    return float(verb_count) / len(doc)


logger = logging.getLogger('rt_over_96H')

""" LOGIN RT """


def login_rt_request(cfg):
    tracker = rt.Rt(str(cfg['request_tracker_db']['address']), str(cfg['request_tracker_db']['user']),
                    str(cfg['request_tracker_db']['password']))
    tracker.login()
    return tracker


""" READ CONFIG FILE """


def read_config(path):
    with open(path, 'r') as stream:
        results = yaml.safe_load(stream)
    return results


def get_tag(conn_tracker, id):
    ticket = conn_tracker.get_ticket(id)
    cf_tag = ticket['CF.{ebi-ega-admin-keywords}']
    return cf_tag

def read_open_dataset():
    # Using readlines()
    dataset = open('../data/test.csv', 'r')
    result = dataset.readlines()
    return result

""" Clean message body """


def clean_query_body(original_message):
    # regex sub
    # urls
    message = re.sub(r'http\S+', '', original_message)
    message = re.sub(r'www\S+', '', original_message)
    # ega accessions
    message = re.sub(r'ega\S+', '', message)
    message = re.sub(r'EGAS\S+', '', message)
    # numbers
    message = re.sub(r'[0-9]+', '', message)
    # months
    message = re.sub(
        '(january|february|march|april|may|june|july|august|september|october|november|december)\s\d{2}\s\d{4}', ' ',
        message)
    # dates
    message = re.sub('^(?:(?:[0-9]{2}[:\/,]){2}[0-9]{2,4}|am|pm)', '', message)
    # common EGA words
    message = re.sub(r'ega', '', message)
    message = re.sub(r'website', '', message)
    message = re.sub(r'could', '', message)
    message = re.sub(r'would', '', message)
    message = re.sub(r'please', '', message)
    message = re.sub(r'needed', '', message)
    message = re.sub(r'happy', '', message)
    message = re.sub(r'assist', '', message)
    message = re.sub(r'help', '', message)
    message = re.sub(r'like', '', message)
    message = re.sub(r'dear', '', message)
    message = re.sub(r'kind', '', message)
    message = re.sub(r'regards', '', message)
    message = re.sub(r'thanks', '', message)
    message = re.sub(r'free', '', message)
    message = re.sub(r'contact', '', message)
    message = re.sub(r'us', '', message)
    message = re.sub(r'we', '', message)
    message = re.sub(r'ask', '', message)
    message = re.sub(r'attach\S+', '', message)
    message = re.sub(r'query', '', message)
    message = re.sub(r'message', '', message)

    results = convert(message, .9)

    ###### CONVERT RESULTS TO STRING
    if not results:
        stop = set(stopwords.words('english') + list(string.punctuation))
        stop_words_removed = [i for i in word_tokenize(message.lower()) if i not in stop]
        return stop_words_removed
    else:
        stop = set(stopwords.words('english') + list(string.punctuation))
        for i in range(len(results)):
            results[i] = results[i].lower()
        target = ' '.join(results)
        stop_words_removed = [i for i in word_tokenize(target) if i not in stop]
        return stop_words_removed


""" First user query  """
def get_initial_query(conn_tracker, id):
    history = conn_tracker.get_history(id)
    initial_query = history[0]
    message = initial_query['Content']
    return message

"""Gets all open tickets"""
def get_all_open_tickets(conn_tracker):
    tickets = conn_tracker.search(Queue='sanger.ac.uk: ebi-ega-helpdesk', Status='open')
    return tickets

"""Gets all resolved tickets"""
def get_all_resolved_tickets(conn_tracker):
    tickets = conn_tracker.search(Queue='sanger.ac.uk: ebi-ega-helpdesk', Status='resolved', Created='' )
    return tickets

""" GETS HISTORY """
def get_history(conn_tracker, id):
    history = conn_tracker.get_history(id)
    for transaction in history:
        creator = (transaction['Creator']).lower().strip()
        rt_system = "rt_system"
        print(creator, rt_system)
        if creator == rt_system:
            pass
        if str(transaction['Content']) == 'This transaction appears to have no content':
            pass
        else:
            print(transaction['Content'])

    # content = history['Content']
    return history


""" GETS LIST OF TARGET TICKETS, REPLIES AND UPDATE TICKET STATUS """


def get_target_tickets(conn_tracker):
    tickets = conn_tracker.search(Queue='sanger.ac.uk: ebi-ega-helpdesk', Status='new', LastUpdated__lt='-1 day')
    for ticket in tickets:
        id = ticket['id'][7:]  # get ticket id
        logger.info("RT: " + str(id) + " updated")  # info
        print("ID:" + id)  # print ticket ID to console
        get_history(conn_tracker, id)
        # reply(conn_tracker, id)  # replies to ticket with static message
        # change_status(conn_tracker, id)  # changes ticket status to new
    return tickets


""" VERIFIES CONNECTION TO DB's  """


def create_dataset(cfg):
    conn_tracker = None
    try:
        conn_tracker = login_rt_request(cfg)  # conn to request Tracker
        if conn_tracker:  # has required connections
            header = ['ticket', 'tag', 'message']
            with open('../data/test.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(header)
                tickets = get_all_open_tickets(conn_tracker)
                #tickets = get_all_resolved_tickets(conn_tracker)
                for ticket in tickets:
                    id = ticket['id'][7:]  # get ticket id
                    print(id)
                    try:
                        initial_query = get_initial_query(conn_tracker, id)
                        print(initial_query)
                        clean_message = clean_query_body(initial_query)
                        print("#########################################################")
                        print(clean_message)
                        print("#########################################################")
                        cf_tag = get_tag(conn_tracker, id)
                        print(cf_tag)
                        id_tag_message = [id, cf_tag, clean_message] # CLEAN message
                        #id_tag_message = [id, cf_tag, initial_query] #  raw message
                        writer.writerow(id_tag_message)
                    except Exception as e:
                        logger.error("Error: {}".format(e))
        else:
            logger.debug("RT is not available")
            logger.info("RT create dataset ended")
    except psycopg2.DatabaseError as e:
        logger.warning("Error:{} ".format(e))
        raise RuntimeError('error') from e
    finally:
        if conn_tracker:
            conn_tracker.logout()
            logger.debug("Request Tracker connection closed")


""" MAIN """
def run():
    try:
        # configure logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s [in %(pathname)s:%(lineno)d]'
        logging.basicConfig(format=log_format)
        # read config file
        cfg = read_config("../bin/config.yml")
        # execute main function
        create_dataset(cfg)
    except Exception as e:
        logger.error("Error: {}".format(e))
        sys.exit(-1)


if __name__ == '__main__':
    ## cue the music
    run()
