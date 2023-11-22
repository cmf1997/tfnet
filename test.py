import csv

tf_name_seq = {
    "HNF4A": "--AGVKNHMYS-QCNQCLFR---AGKK------------",
    "SPI1": "-RDKSHKEALA---KKMMLRNYGTGVKKY----------",
    "GABPA": "-QERLQPELVA---PTMLLRYYYGDICKY----------",
    "GATA3": "N----------------------------ANGDPY----",
    "REST": "HMDNAQVHLN-Y------DRSNFKHVEL-KSKH------",
    "YY1": "HVPQITH--------------------------------",
    "ZNF143": "HVTSVIHYVCT-------EYSSLKHHVV-RTAH------",
    "CTCF": "HMYQIQKHFH-TKSKQHYERYALQHQKS-KRYHFF----",
    "EGR1": "HIDRTIH--------------------------------",
    "TCF7L2": "FM----------EMKVVLAAIN---------------L-",
    "E2F1": "LT-----------------------RFL---------VL",
    "E2F6": "YL-----------------------KFM---------VI",
    "SRF": "------------FILRRS---TGMKKAY-----------",
    "TCF12": "----RMANNARRRD-----AFKEGRCQLT---LLQ----",
    "MAX": "----KAHHNALRDH-----SFHSRDV-P-----------",
    "MYC": "----KRTHNVLRNE-----SFFARDI-PP---VVKELN-",
    "NANOG": "-------------------------KTR---VFSDLNN-",
    "RFX5": "YDETIEIFP--IRLGQS----SGRRKT------------",
    "FOXA1": "-----------------------------PPYSY-ASYA",
    "FOXA2": "-----------------------------PPYSY-SNFP",
    "STAT3": "YQI-------------VGFNILGNTKVMALPVVVNNKSA",
    "TEAD4": "QSIIYRN----LYITGKKSHIQVARKAR-----------",
    "ARID3A": "------------------EFLDDFS---NMLYVLE--Y-",
    "MAFK": "-------VTRLQTLGYAITQKEEERQRVRSKYEA-----",
    "CREB1": "-------EAARRLMEAAREYVKCENRVAEDLYCH-----",
    "CEBPB": "---------SDYREIAVDMRNLEQHKVLKNLFK------",
    "ATF3": "-------EEDEKREIAANEKTECQKLPR-----------",
    "JUND": "-------QERIARLIAAKERISREEKVKSQKVL------",
    "ATF7": "--NNKKHDPDERLEAAAQLWVSSEKKAENQLLL------",
}
with open("data/tf_pseudosequences.txt", "w") as fp:
    writer = csv.writer(fp, delimiter="\t")
    for key, value in tf_name_seq.items():
        writer.writerow([key, value])
