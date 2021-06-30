'''data.py
Reads CSV files, stores data, access/filter data by variable name
YOUR NAME HERE
CS 251 Data Analysis and Visualization
Spring 2021
'''

import numpy as np
import csv
import os
import datetime



class Data:

    ''' defining a list of data types over all kinds of data sets'''

    dataTypes = ['string','enum','numeric','date']

    def __init__(self, filepath=None, headers=None, data=None, header2col=None, dataFields=None, dataDict=None, rowsToPrint=5):
        '''Data object constructor

        Parameters:
        -----------



        filepath: str or None. Path to data .csv file'''
        self.filepath = filepath




        '''
        headers: Python list of strings or None. List of strings that explain the name of each
            column of data.'''
        if headers != None:
            self.headers = headers
        else:
            self.headers = []




        '''
        data: ndarray or None. shape=(N, M).
        N is the number of data samples (rows) in the dataset and M is the number of variables
        (cols) in the dataset.
        2D numpy array of the dataset’s values, all formatted as floats.'''

        try:
            if data == None:
                self.data = []
        except:
            self.data = data




        ''' dictonary to store all data'''
        if dataDict != None:
            self.dataDict = dataDict
        else:
            self.dataDict = {}


        '''
        header2col: Python dictionary or None.
                Maps header (var str name) to column index (int).
                Example: "sepal_length" -> 0
        '''
        if header2col != None:
            self.header2col = header2col
        else:
            self.header2col = {}

        ''' dataFields: Python dictionary or None
                this holds a dicytionary of all the data fields in the data
                the key is the name (headers) of the data and the value is the data type             
        '''
        if dataFields != None:
            self.dataFields = dataFields
        else:
            self.dataFields = {}


        ''' rowsToPrint:
            must be an int if it is not defualt of 5 rows is changed
            if not an int return error
        
        '''

        if isinstance(rowsToPrint, int):
            self.rowsToPrint = rowsToPrint
        else:
            print(f'ERROR: rowsToPrint must be an integer\n{rowsToPrint} is not an int\n5 rows will be printed for Data Object')
            self.rowsToPrint = 5


        '''
        TODO:
        - If `filepath` isn't None, call the `read` method.
        '''

        if self.filepath != None:
            self.read(filepath=self.filepath)
        else:
            # print("WARNING: There is no File_Path")
            pass


    ''' this is a helper function to see if all objects in the array
    contain a string
    
    used in:
        read
        
    returns:
        true if array has all strings
    '''
    def arrayHasAllStrings(self, array_to_be_checked):

        for object in array_to_be_checked:
            try:
                float(object)
                return False
            except:
                pass

        return True





        ''' this is a helper function to see if all objects in the array
            contain a a proper data type heading

            returns:
                true if array has all proper data type headings
            '''

    def hasDataTypeString(self, array_to_be_checked):

        for object in array_to_be_checked:
            if object not in self.dataTypes:
                print(f"ERROR: all Data Needs Headers!\nEXITING PROGRAM!!!") #not sure if this is the correct error code
                exit()
                return False
        return True









    def read(self, filepath, testType=0):


        if filepath == None:
            return None
        else:
            try:
                '''thanks to Corey Schafer for this video teaching me clearly how to use csv module
                 https://www.youtube.com/watch?v=q5uM4VKywbA&t'''

                csvFilePath  = f"{filepath}"
                self.filepath = csvFilePath
                with open(csvFilePath, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)

                    ''' get the number of data colums and row for the data array'''

                    csv_reader_list = list(csv_reader)
                    DataRowLen = len(csv_reader_list) - 2

                    '''====================================='''

                    #set up empty data array
                    num_colums_in_data = 0
                    data_indexes = []
                    for index_of_data, data_point in enumerate(csv_reader_list[3]):

                        # thanks to https://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python
                        # for info on how to test for float
                        try:
                            float(data_point)
                            num_colums_in_data += 1
                            data_indexes.append(index_of_data)
                        except ValueError:
                            pass


                    self.data = np.zeros((DataRowLen, num_colums_in_data))
                    for line_number , line in enumerate(csv_reader_list):

                        #dont like how doing

                        for colNumberL, object in enumerate(line):
                            #strip or replace
                            #spaceStripedObject = object.strip(' ')
                            spaceStripedObject = object.replace(' ', '')
                            csv_reader_list[line_number][colNumberL] = spaceStripedObject

                        ''' this block of code gets the headers of the DATA
                            and set header2col
                        '''
                        if line_number == 0:

                            data_index = 0
                            for colNum , header in enumerate(line):
                                if colNum in data_indexes:
                                    self.headers.append(header)
                                    self.header2col[header] = data_index
                                    data_index += 1


                        ''' This block of code gets the data types for each headers'''
                        #array to hold all the data types
                        dataTypesForData = []

                        if line_number == 1:
                            if (self.hasDataTypeString(line)):
                                for colNumber, dataType in enumerate(line):
                                    if colNumber in data_indexes:
                                        self.dataFields[self.headers[data_indexes.index(colNumber)]] = dataType
                            else:

                                ''' if it is test case 0 (the default to pass class tests)'''
                                if testType == 0:
                                    #for header in self.headers:
                                    # I want to add data types here:
                                    print("ERROR: Data Needs Data Type Headers")
                                    return None
                                elif testType == 1:
                                    dataLists = [{self.headers[rNum] : []} for rNum in range(DataRowLen)]
                                    for data_line_number , data_line in enumerate(csv_reader_list[1:]):
                                        for data_col, dataInfo in enumerate(data_line):
                                            if data_col in data_indexes:
                                                dataLists[data_col][self.headers[data_line_number]].append[data]


                                    print(dataLists)






                        ''' this block of code is to set self.dataDict'''
                        if line_number >= 2:

                            "loop through the data"
                            for cNum ,data in enumerate(line):
                                if cNum in data_indexes:
                                    dataType_for_col = list(self.dataFields.values())[data_indexes.index(cNum)]
                                    if dataType_for_col == 'numeric':
                                        self.data[line_number-2][data_indexes.index(cNum)] = data

                                #self.dataDict[line_number].append({f"Header: {self.headers[line_number]} Type: {self.dataFields[self.headers[line_number]]}" : data})

                self.data = np.array(self.data)
            except IOError:
                print("Error: CSV File with this File-Path does not seem to exist.")




    # #helper method for to string method
    # def mainPrintHelper(self
    #
    #

    def __str__(self):
        '''toString method

        (For those who don't know, __str__ works like toString in Java...In this case, it's what's
        called to determine what gets shown when a `Data` object is printed.)

        can modify rows int to call number of rows

        Returns:
        -----------
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.
        '''

        full_data_string = f"\n-------------------------------\n{self.filepath} ({np.shape(self.data)[0]}x{np.shape(self.data)[1]})"
        full_data_string += f"\nHeaders:\n"

        #get all headers fo hearders line
        heads_string = ''
        for header in self.headers:
            heads_string += f'\t{header}'

        full_data_string +=heads_string
        full_data_string += f"\n-------------------------------"
        if self.rowsToPrint <= self.data.shape[0]:
            full_data_string += f"\n Showing first {self.rowsToPrint}/{np.shape(self.data)[0]} rows."

        for row in self.data[:self.rowsToPrint]:
        # thank you to https://stackoverflow.com/questions/59956496/f-strings-formatter-including-for-loop-or-if-conditions
        # to learn how to loop f strings
            data_sample_row_string = "\t".join(f"{data_point:4}" for data_point in row)
            full_data_string += f"\n{data_sample_row_string}"
            #full_data_string += ("\n".join(f"\t{data_point}" for data_point in self.data[0:self.rowsToPrint]))
        # print(full_data_string)

        # dataFieldsStrings = (f"\n{list(self.dataFields.keys())}\n{list(self.dataFields.values())}")
        # dataStrings = ''
        # for row in self.data[:self.rowsToPrint]:
        #     dataStrings += f'\n{row}'
        # full_data_string += dataFieldsStrings + dataStrings

        return (full_data_string)

    def get_headers(self):
        '''Get method for headers

        Returns:
        -----------
        Python list of str.
        '''
        return list(self.headers)

    def get_mappings(self):
        '''Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        '''
        return dict(self.header2col)

    def get_num_dims(self):
        '''Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        '''
        return len(self.headers)


    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        '''
        return np.shape(self.data)[0]

    def get_sample(self, rowInd):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        '''
        return self.data[rowInd]

    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers`
            list.
        '''

        index_list = []
        for header in headers:
            if header.replace(" ", '') in self.header2col:
                index_list.append(self.header2col[header.replace(" ", '')])
            else:
                print(f'\n ERROR: Header "{header.replace(" ", "")}" not in data object')

        return index_list

    def get_all_data(self):
        '''Gets a copy of the entire dataset

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
            NOTE: This should be a COPY, not the data stored here itself.
            This can be accomplished with numpy's copy function.
        '''
        return self.data.copy()

    def head(self, numInHead=5):
        '''Return the 1st five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        '''
        return (self.data[:numInHead])

    def tail(self, numInTail=5):
        '''Return the last five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        '''
        return (self.data[-numInTail:])

    def limit_samples(self, start_row, end_row):
        '''Update the data so that this `Data` object only stores samples in the contiguous range:
            `start_row` (inclusive), end_row (exclusive)
        Samples outside the specified range are no longer stored.

        (Week 2)

        '''

        newDataArray = self.data[start_row:end_row][:]
        self.data = newDataArray

    def select_data(self, headers, rows=[]):
        '''Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified
        by the `rows` list.

        (Week 2)

        For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return
        column #2 of self.data. If rows is not [] (say =[0, 2, 5]), then we do the same thing,
        but only return rows 0, 2, and 5 of column #2.

        Parameters:
        -----------
            headers: Python list of str. Header names to take from self.data
            rows: Python list of int. Indices of subset of data samples to select.
                Empty list [] means take all rows

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.

        Hint: For selecting a subset of rows from the data ndarray, check out np.ix_
        '''



        # check if there are rows or not
        if len(rows) == 0:


            # Initilize list to hold all the positions
            header_pos_list = []

            ''' First Loop Through all the headers passed in'''
            for header_to_check in headers:
                # clean header
                spaceStripedObject = str(header_to_check).replace(' ', '')

                if spaceStripedObject in self.headers:
                    header_pos_list.append(self.header2col[spaceStripedObject])
                else:
                    print(f'\nERROR: Header "{spaceStripedObject}" is not in Data Object {self.headers}')



            # Create the return array:
            return_array = self.data[np.ix_(np.arange(self.data.shape[0]), header_pos_list)]


            return return_array

        # if there are rows
        else:
            # Initilize list to hold all the positions
            header_pos_list = []

            ''' First Loop Through all the headers passed in'''
            for header_to_check in headers:
                # clean header
                spaceStripedObject = header_to_check.replace(' ', '')

                if spaceStripedObject in self.headers:
                    header_pos_list.append(self.header2col[spaceStripedObject])
                else:
                    print(f'\nERROR: Header "{spaceStripedObject}" is not in Data Object')

            # create an array manipulated in proper way to make ix_ mehtod easier to use
            re_arranged_data = np.fliplr(np.rot90(self.data))
            sub_index_array = np.ix_(rows, header_pos_list)

            # Create the return array:
            return_array = self.data[sub_index_array]

            return return_array





# THIS IS AN EXTENSION FOR AN EXTENDED CLASS THAT TAKE ALL DATA TYPES
class AllData(Data):
    #dictionary to hold types with proper data type string
    dataTypes = {'string' : 'U25', 'enum': 'U', 'numeric': 'f', 'date': 'M'}

    def __init__(self, filepath=None, headers=None, data=None, header2col=None, dataFields=None, dataDict=None,
                 rowsToPrint=5):
        '''Data object constructor

        Parameters:
        -----------



        filepath: str or None. Path to data .csv file'''
        self.filepath = filepath

        '''
        headers: Python list of strings or None. List of strings that explain the name of each
            column of data.'''
        if headers != None:
            self.headers = headers
        else:
            self.headers = []

        '''
        data: ndarray or None. shape=(N, M).
            N is the number of data samples (rows) in the dataset and M is the number of variables
            (cols) in the dataset.
            2D numpy array of the dataset’s values, all formatted as STRINGS!!!!!!!.
    
        '''
        if data != None:
            self.data = data
        else:
            self.data = []



        '''
        header2col: Python dictionary or None.
                Maps header (var str name) to column index (int).
                Example: "sepal_length" -> 0
        '''
        if header2col != None:
            self.header2col = header2col
        else:
            self.header2col = {}

        ''' dataFields: Python dictionary or None
                this holds a dicytionary of all the data fields in the data
                the key is the name (headers) of the data and the value is the data type             
        '''
        if dataFields != None:
            self.dataFields = dataFields
        else:
            self.dataFields = {}

        ''' rowsToPrint:
            must be an int if it is not defualt of 5 rows is changed
            if not an int return error

        '''

        if isinstance(rowsToPrint, int):
            self.rowsToPrint = rowsToPrint
        else:
            print(
                f'ERROR: rowsToPrint must be an integer\n{rowsToPrint} is not an int\n5 rows will be printed for Data Object')
            self.rowsToPrint = 5


        if self.filepath != None:
            self.read(filepath=self.filepath)
        else:
            print("WARNING: There is no File_Path")





    #helper function to create proper data type for array
    def createDataType(self):
        data_field_headers = self.headers
        data_fields_types = [self.dataTypes[typ] for typ in self.dataFields.values()]
        datatype_dict = {'names' : data_field_headers, 'formats': data_fields_types}
        return np.dtype(datatype_dict)



    def read(self, filepath, testType=0):


        if filepath == None:
            return None
        else:
            try:
                '''thanks to Corey Schafer for this video teaching me clearly how to use csv module
                 https://www.youtube.com/watch?v=q5uM4VKywbA&t'''

                csvFilePath  = f"{filepath}"
                self.filepath = csvFilePath
                with open(csvFilePath, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)

                    ''' get the number of data colums and row for the data array'''

                    csv_reader_list = list(csv_reader)
                    sample_ammount = len(csv_reader_list) - 2
                    data_field_ammount = len(csv_reader_list[3])

                    data_array_lists = []

                    '''====================================='''

                    for line_number , line in enumerate(csv_reader_list):

                        #dont like how doing

                        for colNumberL, object in enumerate(line):
                            #strip or replace
                            #spaceStripedObject = object.strip(' ')
                            spaceStripedObject = object.replace(' ', '')
                            csv_reader_list[line_number][colNumberL] = spaceStripedObject

                        ''' this block of code gets the headers of the DATA
                            and set header2col
                        '''
                        if line_number == 0:
                            if(self.arrayHasAllStrings(line)):

                                for colNum , header in enumerate(line):
                                    self.headers.append(header)
                                    self.header2col[header] = colNum

                            else:
                                print("ERROR: Data Needs Headers")
                                return None


                        ''' This block of code gets the data types for each headers'''
                        #array to hold all the data types
                        dataTypesForData = []

                        if line_number == 1:
                            if (self.hasDataTypeString(line)):
                                for colNumber, dataType in enumerate(line):
                                    self.dataFields[self.headers[(colNumber)]] = dataType
                            else:

                                ''' if it is test case 0 (the default to pass class tests)'''
                                if testType == 0:
                                    #for header in self.headers:
                                    # I want to add data types here:
                                    print("ERROR: Data Needs Data Type Headers")
                                    return None
                                # elif testType == 1:
                                #     dataLists = [{self.headers[rNum] : []} for rNum in range(sample_ammount)]
                                #     for data_line_number , data_line in enumerate(csv_reader_list[1:]):
                                #         for data_col, dataInfo in enumerate(data_line):
                                #             if data_col in data_indexes:
                                #                 dataLists[data_col][self.headers[data_line_number]].append[data]
                                #
                                #
                                #     print(dataLists)


                        # #create the self.data structured array with proper data type
                        # if line_number == 2:




                        ''' this block of code is to set self.dataDict'''
                        if line_number >= 2:

                            ##question https://stackoverflow.com/questions/11309739/store-different-datatypes-in-one-numpy-array

                            sample_data_list =[]
                            "loop through the data"
                            for cNum ,data in enumerate(line):
                                dataType_for_col = list(self.dataFields.values())[cNum]
                                if dataType_for_col == 'numeric':
                                    sample_data_list.append(float(data))
                                elif dataType_for_col == 'date':
                                    sample_data_list.append(datetime.datetime.strptime(data,"%m/%d/%Y"))
                                else:
                                    sample_data_list.append(data)

                            #self.dataDict[line_number].append({f"Header: {self.headers[line_number]} Type: {self.dataFields[self.headers[line_number]]}" : data})
                            sample_tup = tuple(data for data in sample_data_list)
                            data_array_lists.append(sample_tup)

                    array_data_types = self.createDataType()
                    self.data = np.array(data_array_lists, dtype = array_data_types)
            except IOError:
                print("Error: CSV File with this File-Path does not seem to exist.")


    #helper function to create data
    def selected_data_Create(self, headers_Same_type, sub_array):

        headerDict = {}
        for header in headers_Same_type:
            headerDict[header] = []

        header_type_array = sub_array[headers_Same_type]
        for sample in header_type_array:
            for data_index, data in enumerate(sample):
                headerDict[headers_Same_type[data_index]].append([data])
        return_array = np.flip(np.rot90(np.array(list(headerDict.values())).reshape(len(headers_Same_type), len(sub_array))),0)







        return return_array


    def select_data(self, headers, rows=[]):
        if len(rows) == 0:


            # Initilize list to hold all the positions
            header_pos_list = []
            clean_header_list = []
            ''' First Loop Through all the headers passed in'''
            for header_to_check in headers:
                # clean header
                spaceStripedObject = str(header_to_check).replace(' ', '')
                clean_header_list.append(spaceStripedObject)
                if spaceStripedObject in self.headers:
                    header_pos_list.append(self.header2col[spaceStripedObject])
                else:
                    print(f'\nERROR: Header "{spaceStripedObject}" is not in Data Object {self.headers}')



            sub_array = np.ix_(list(np.arange(len(self.data))))
            # Create the return array:
            return_array = self.data[sub_array]

            #check for all the different data types
            date_time_array = []
            string_array = []
            numeric_array = []

            for header in clean_header_list:
                header_type = self.dataFields[header]
                if header_type == 'numeric':
                    numeric_array.append(header)
                elif header_type == 'date':
                    date_time_array.append(header)
                #else if a string
                else:
                    string_array.append(header)



            #check for what arrays to return and make them calling helper function
            return_arrays_list = []
            if len(numeric_array) > 0:
                return_arrays_list.append(self.selected_data_Create(numeric_array,return_array))

            if len(string_array) > 0:
                return_arrays_list.append(self.selected_data_Create(string_array,return_array))

            if len(date_time_array) > 0:
                return_arrays_list.append(self.selected_data_Create(date_time_array,return_array))


            #see what to return
            if len(return_arrays_list) == 1:
                return  return_arrays_list[0]
            elif len(return_arrays_list)>1:
                return  return_arrays_list

        # if there are rows
        else:
            # Initilize list to hold all the positions
            header_pos_list = []
            clean_header_list = []

            ''' First Loop Through all the headers passed in'''
            for header_to_check in headers:
                # clean header
                spaceStripedObject = header_to_check.replace(' ', '')
                clean_header_list.append(spaceStripedObject)
                if spaceStripedObject in self.headers:
                    header_pos_list.append(self.header2col[spaceStripedObject])
                else:
                    print(f'\nERROR: Header "{spaceStripedObject}" is not in Data Object')


            sub_index_array = np.ix_(rows)

            # Create the return array:
            return_array = self.data[sub_index_array]

            # check for all the different data types
            date_time_array = []
            string_array = []
            numeric_array = []

            for header in clean_header_list:
                header_type = self.dataFields[header]
                if header_type == 'numeric':
                    numeric_array.append(header)
                elif header_type == 'date':
                    date_time_array.append(header)
                # else if a string
                else:
                    string_array.append(header)

            # check for what arrays to return and make them calling helper function
            return_arrays_list = []
            if len(numeric_array) > 0:
                return_arrays_list.append(self.selected_data_Create(numeric_array,return_array))

            if len(string_array) > 0:
                return_arrays_list.append(self.selected_data_Create(string_array,return_array))

            if len(date_time_array) > 0:
                return_arrays_list.append(self.selected_data_Create(date_time_array,return_array))

            # see what to return
            if len(return_arrays_list) == 1:
                return return_arrays_list[0]
            elif len(return_arrays_list) > 1:
                return return_arrays_list

''' this is a function to test my data object class'''
def main():

    '''test for no filepath name'''
    noFilePathClass = Data()


    ''' test for wrong file path given (the file does not exist)'''
    wrongFilePathClass = Data(filepath = "/datasets/anscombeiasdd")


    ''' test iris csv file'''
    irisDataClass = Data(filepath = "datasets/iris")

    ''' test iris_bad csv file'''
    iris_Bad_Data_Class = Data(filepath="/home/matt/Colby/cs251/ProjectsVenv/share/project1/lab1/datasets/iris_bad.csv")



    ''' test anscombe.csv file '''
    anscombe_Data_Class = Data(filepath="/home/matt/Colby/cs251/ProjectsVenv/share/project1/lab1/datasets/anscombe.csv")




    ''' test test_data_complex.csv file '''
    test_data_complex_Data_Class = Data(filepath="/home/matt/Colby/cs251/ProjectsVenv/share/project1/lab1/datasets/test_data_complex.csv", rowsToPrint=2)



    ''' test test_data_spaces.csv.csv file '''
    test_data_spaces_Data_Class = Data(filepath="/home/matt/Colby/cs251/ProjectsVenv/share/project1/lab1/datasets/test_data_spaces.csv")



    # ''' test the to string method'''
    # print(irisDataClass)
    # print(test_data_complex_Data_Class)


    # ''' testing other methods with different csv files'''
    # print(f"\n{irisDataClass.get_headers()}")
    #
    # print(f"\n{irisDataClass.get_mappings()}")
    #
    # print(f"\n{irisDataClass.get_num_dims()}")
    #
    # print(f"\n{irisDataClass.get_num_samples()}")
    #
    # print(f"\n{irisDataClass.get_sample(34)}")
    #
    # print(f"\n{irisDataClass.get_header_indices([' sepal _length  ', 'color', ' species', 'petal_width'])}")
    #
    # print(f"\n{irisDataClass.head()}")
    # print(f"\n{irisDataClass.tail()}")
    #
    # print(f"\n ORIGINAL DATA")
    #
    # print(f"\n{irisDataClass.get_all_data()}")
    #
    # irisDataClass.limit_samples(1,8)
    #
    # print(f"\n  DATA AFTER LIMIT SAMPLE")
    # print(f"\n{irisDataClass.get_all_data()}")
    #
    # print(f"\n{irisDataClass.select_data([' sepal _length  ', 'color', ' species', 'petal_width'])}")
    #
    # print(f"\n{irisDataClass.select_data([' sepal _length  ', 'color', ' species', 'petal_width'], [1,7,10,23])}")

    # test_filename = 'data/test_data_spaces.csv'
    # test_data = Data(test_filename)
    #
    # one = test_data.select_data(['spaces'])
    # print(f'All data in the "spaces" variable (shape={one.shape}): \n{one}')
    #
    # two = test_data.select_data(['spaces', 'places'])
    # print(f'All data in the "spaces" and "places" variables (shape={two.shape}): \n{two}')

    ''' test iris csv file AllData'''
    test_Alldara = AllData('data/iris.csv')
    #{'sepal_length': 'numeric', 'sepal_width': 'numeric', 'petal_length': 'numeric', 'petal_width': 'numeric', 'species': 'string'}
    s_num = test_Alldara.select_data(headers=['sepal_length', "petal_width"])
    s = test_Alldara.select_data(headers=['sepal_length', "species"])
    print(s[0])
    return



if __name__ == "__main__":
    main()