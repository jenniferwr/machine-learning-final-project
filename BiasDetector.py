# Bias detection in nyc jobs dataset
import csv
def loadData():
    with open('nyc-jobs.csv', 'rt') as f:
        reader = csv.reader(f)
        data = list(reader)

    # 2d list to contain all the information about a job
    # jobInfo[1] will contain all info about job at index 1
    # 'Business Title', 'Level', 'Job Category', 'Full-Time/Part-Time indicator', 'Salary Range From', 'Salary Range To', 'Salary Frequency'
    w, h = 7, len(data);
    jobInfo = [[0 for x in range(w)] for y in range(h)]

    # 2d list, stores 'Job Description', 'Minimum Qual Requirements', 'Preferred Skills', 'Additional Information'
    # jobDesc[1] will contain info about job at index 1
    w, h = 4, len(data);
    jobDesc = [[0 for x in range(w)] for y in range(h)]

    # loop to put the data from the larger dataset into our lists
    for x in range(1, len(data)):
        jobInfo[x][0] = data[x][4]
        jobInfo[x][1] = data[x][7]
        jobInfo[x][2] = data[x][8]
        jobInfo[x][3] = data[x][9]
        jobInfo[x][4] = data[x][10]
        jobInfo[x][5] = data[x][11]
        jobInfo[x][6] = data[x][12]


        jobDesc[x][0] = data[x][16]
        jobDesc[x][1] = data[x][17]
        jobDesc[x][2] = data[x][18]
        jobDesc[x][3] = data[x][19]

    return jobInfo, jobDesc


def main():
    jobInfo, jobDesc = loadData()

main()
