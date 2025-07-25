import json
import networkx as nx
from Flowsheet_Class.flowsheet import Flowsheet

# Load the JSON dataset
with open('train_data_sfiles/train_data_10k.json', 'r') as file:
    dataset = [json.loads(line.strip()) for line in file]


# Extract all PID strings from dataset
tot = 0
corr = 0 
pid_strings = [entry.get("PID", "") for entry in dataset]
# pfd_string = [entry.get("PFD", "") for entry in dataset]
def test_conversion(sfile):
    '''Case 1 : sfiles(pid)--> graph --> sfiles(pid)
        Case 2 : sfiled(pid)--> sfiles(pfd) --> graph --> sfiles(pfd)
    '''
    global tot,corr
    sfilesctrl1 = sfile
    graph = nx.DiGraph()
    flowsheet = Flowsheet()
    flowsheet.state = graph
    flowsheet.create_from_sfiles(sfilesctrl1, overwrite_nx=True)
    flowsheet.convert_to_sfiles()
    sfilesctrl2 = flowsheet.sfiles

    sfiles = flowsheet.convert_sfilesctrl_to_sfiles()
    flowsheet.sfiles = sfiles
    flowsheet.create_from_sfiles(sfiles, overwrite_nx=True)
    flowsheet.convert_to_sfiles()
    sfiles2 = flowsheet.sfiles
    tot += 1 
    print(tot)
    if sfilesctrl1 == sfilesctrl2 and sfiles == sfiles2:
        # print(
        #     "-----------------------------------------------------------------------------------------------------")
        # print("Test case ",tot)
        # print("Conversion back successful")
        # print("SFILESctrl: ", sfilesctrl1)
        # print("SFILES: ", sfiles)
        corr +=1
    else:
        print(
            "-----------------------------------------------------------------------------------------------------")
        print("Test case ", tot)
        print("Conversion back produced a different SFILES string. Input:", sfilesctrl1, "Output:", sfilesctrl2)
        print("Conversion back produced a different SFILES string. Input:", sfiles, "Output:", sfiles2)

    # self.assertEqual(sfilesctrl1, sfilesctrl2, "Not correct!")
    # self.assertEqual(sfiles, sfiles2, "Not correct!")

if __name__ == "__main__":
    for pid_string,pfd_string in pid_strings:
        test_conversion(pid_string)   
    accuracy = round((corr / tot) * 100, 3)
    print("Accuracy : ",accuracy)
    print(tot-corr)