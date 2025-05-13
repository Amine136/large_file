from utils import _generate_and_save, retrieval_qa
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyBD1ssf_jhmQblYIiarFKuFOYg5fZCKLvI"




#model1

example1 = """Âge: 78
            Sexe: F
            Hématies (GR, RBC): 3.84 (Previous: 3.66) [Ref: 3.8-5.2]
            Hémoglobine (Hb): 12.7 (Previous: 12.1) [Ref: 11.8-15]
            Hématocrite (Hte): 37.8 (Previous: 35.9) [Ref: 35-45]
            Volume globulaire moyen (VGM, MCV): 98.4 (Previous: 98) [Ref: 83-98]
            Indice de distribution des globules rouges (IDR, RDW): 17 (Previous: 17) [Ref: 11-17]
            Concentration corpusculaire moyenne en hémoglobine (CCMH, MCHC): 34 (Previous: 34) [Ref: 32-36]
            Teneur corpusculaire moyenne en hémoglobine (TCMH, MCH): 33.1 (Previous: 33.1) [Ref: 27-33]
            Plaquettes (PLT): 423 (Previous: 411) [Ref: 175-380]
            Volume plaquettaire moyen (VPM) (MPV): 8.5 (Previous: 8.2) [Ref: 7.6-10.4]
            Leucocytes (GB, WBC): 5.7 (Previous: 6.2) [Ref: 3.8-9.1]
            Polynucélaires neutrophiles: 61.1 (Previous: 63.8) [Ref: 50-63]
            Polynucélaires neutrophiles (PNN): 3.45 (Previous: 3.92) [Ref: 1.9-5.7]
            Polynucléaires éosinophiles: 2.2 (Previous: 2.7) [Ref: 1.1-5.7]
            Polynucléaires éosinophiles (PNE): 0.13 (Previous: 0) [Ref: 0.04-0.52]
            Polynucléaires basophiles: 0.8 (Previous: 1.1) [Ref: 0-1]
            Polynucléaires basophiles (PNB): 0.04 (Previous: 0) [Ref: 0-0.1]
            Lymphocytes: 29.6 (Previous: 22.6) [Ref: 26-43]
            Lymphocytes (LYMPHS): 1.67 (Previous: 1) [Ref: 1-3.9]
            Monocytes: 6.3 (Previous: 9.8) [Ref: 5-7]
            Monocytes (MONOS): 0.36 (Previous: 0.61) [Ref: 0.2-0.6]"""

_generate_and_save("reports/analyse.json", example1) #path + data



#model2

question = "expectation-induced pain inform the design of non-pharmacological pain treatments?"

print(retrieval_qa.run((question,question)))
