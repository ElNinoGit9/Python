import numpy as np
import random
from win32com.client import Dispatch
from textblob import TextBlob

class SpeechAIClass:
    def __init__(self):

            self.greetings = ("hello", "hi", "greetings", "what's up")
            self.greetings_respons = ("hello", "hi", "greetings", "what's up")
            self.general_respons = ('okay?', 'interesting', 'hmm')

            self.Speech('Welcome')

    def Respond(self, sentence):

            cleaned = self.Parse_sentence(sentence)
            parsed = TextBlob(cleaned)

            pronoun, noun, adjective, verb = self.Divide_sentence(parsed)

            print(pronoun)
            print(noun)
            print(adjective)
            print(verb)

            respons = self.CheckForGreeting(parsed)

            if respons is None:
                    respons = self.ConstructRespons(pronoun, noun, adjective, verb)

            self.Speech(respons)

    def CheckForGreeting(self, sentence):

            if sentence.lower() in self.greetings:
                return random.choice(self.greetings_respons)

    def ConstructRespons(self, pronoun, noun, adjective, verb):

        respons = []

        if not pronoun and not noun and not verb and not adjective:
            respons = random.choice(self.general_respons)

        elif not noun and not verb and not adjective:
            respons = self.ConstructPronounRespons(pronoun)

        elif not pronoun and not verb and not adjective:
            respons = self.ConstructNounRespons(noun)

        elif not noun and not pronoun and not adjective:
            respons = self.ConstructVerbRespons(verb)

        elif not noun and not pronoun and not verb:
            respons = self.ConstructAdjectiveRespons(adjective)

        elif not noun and not adjective:
            respons = self.ConstructPronounVerbRespons(pronoun, verb)

        elif not pronoun and not adjective:
            respons = self.ConstructNounVerbRespons(noun, verb)

        elif not pronoun and not verb:
            respons = self.ConstructNounAdjectiveRespons(noun, adjective)

        elif not verb and not adjective:
            respons = self.ConstructPronounNounRespons(pronoun, noun)

        elif not adjective:
            respons = self.ConstructPronounNounVerbRespons(pronoun, noun, verb)

        else:
            respons = self.ConstructResponsAll(pronound, noun, adjective, verb)

        return respons

    def ConstructPronounRespons(self, pronoun):

        if pronoun is 'You':
            respons = 'You what?'
        elif pronoun is 'I':
            respons = 'Me what?'
        else:
            respons = '?'

        return respons

    def ConstructVerbRespons(self, verb):

        respons = [verb + '?']

        return respons

    def ConstructAdjectiveRespons(self, adjective):

        respons = ['what is ' + adjective + '?']

        return respons

    def ConstructNounRespons(self, noun):

        nounRespons = ('Oh, I like ', 'Im sorry, I dont like ')
        respons = [random.choice(nounRespons) + noun]

        return respons

    def ConstructNounVerbRespons(self, noun, verb):

        respons = ('Is the ' + noun + ' ' + verb + '?')

        return respons

    def ConstructNounAdjectiveRespons(self, noun, adjective):

        respons = ('Is the ' + noun + ' really ' + adjective, 'Are you sure that the ' + noun + ' is ' + adjective)

        return random.choice(respons)

    def ConstructPronounVerbRespons(self, pronoun, verb):

        respons = (pronoun + ' ' + verb + ' ' + 'what?')

        return respons

    def ConstructResponsAll(self, pronoun, noun, adjective, verb):



        return respons

    def Parse_sentence(self, sentence):

        cleaned = []
        words = sentence.split(' ')
        for w in words:
            if w == 'i':
                w = 'I'
            if w == "i'm":
                w = "I'm"
            cleaned.append(w)

        return ' '.join(cleaned)


    def Divide_sentence(self, words):

                pronoun = None
                noun = None
                adjective = None
                verb = None

                for w in words.sentences:
                    pronoun = self.Find_pronoun(w)
                    noun = self.Find_noun(w)
                    adjective = self.Find_adjective(w)
                    verb = self.Find_verb(w)

                return pronoun, noun, adjective, verb

    def Find_pronoun(self, words):

        pronoun = None

        for word, part_of_speech in words.pos_tags:
            # Disambiguate pronouns
            if part_of_speech == 'PRP' and word.lower() == 'you':
                pronoun = 'I'
            elif part_of_speech == 'PRP' and word == 'I':
                # If the user mentioned themselves, then they will definitely be the pronoun
                pronoun = 'You'
        return pronoun

    def Find_noun(self, words):

        noun = None

        for w, p in words.pos_tags:
            if p == 'NN':
                noun = w
                break

        return noun

    def Find_adjective(self, words):

        adjective = None

        for w, p in words.pos_tags:
            if p == 'JJ':
                adjective = w
                break

        return adjective

    def Find_verb(self, words):

        verb = None
        pos = None
        for word, part_of_speech in words.pos_tags:
            if part_of_speech.startswith('VB'):  # This is a verb
                verb = word
                pos = part_of_speech
                break

        return verb

    def Speech(self, text):

            speak = Dispatch("SAPI.SpVoice")

            speak.Speak(text)

SP_Class = SpeechAIClass()

SP_Class.Respond('I try')
