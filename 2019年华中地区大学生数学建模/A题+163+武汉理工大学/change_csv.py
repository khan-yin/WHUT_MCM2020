#-*-coding:utf-8 -*-
import csv

def clear_filter(line):
    '''对每一行的数据进行空格清理'''
    str_list = filter(None,line.strip('\n').split(" "))
    return str_list

def write_csv(txt_str,csv_str):
    '''将文件转化为cvs中'''
    with open(csv_str, 'a') as csvfile:
        spamwriter = csv.writer(csvfile, dialect='excel')
        with open(txt_str, 'r') as filein:
            for line in filein:
                line_list = clear_filter(line)
                spamwriter.writerow(line_list)

if __name__ == "__main__":
    num = 2000

    for i in range(2014-2000):
        txt_str = r'C:\Users\77526\PycharmProjects\untitled\华中赛\A题附件\CH' + str(num) + 'BST.txt'
        csv_str = r'C:\Users\77526\PycharmProjects\untitled\华中赛\data\new_data.csv'
        write_csv(txt_str,csv_str)
        num+=1
        i+=1