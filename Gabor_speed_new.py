import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib as mpl
from matplotlib import rc, font_manager
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import binary_opening, binary_closing, thin, disk, remove_small_objects, remove_small_holes
from skan import csr
from sklearn.cluster import AgglomerativeClustering
from mpl_toolkits.axes_grid1 import Divider, Size


class GaborLineDetection():

    def __init__(self, path_cross_correlation, debug=False):
        self.path = path_cross_correlation
        self.debug = debug

        self.title = self.path.split('\\')

        if 'PT' in self.title[-3] or 'ST' in self.title[-3]:
            self.title = self.title[-3] + ': ' + self.title[-2]
        elif 'PP' in self.title[-2] or 'SP' in self.title[-2]:
            self.title = self.title[-2]
        elif 'PP' in self.title[-3] or 'SP' in self.title[-3]:
            self.title = self.title[-3]

    def set_size(w, h, ax=None):
        """ w, h: width, height in inches """
        if not ax: ax = plt.gca()
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w) / (r - l)
        figh = float(h) / (t - b)
        ax.figure.set_size_inches(figw, figh)

    def setupParameters(self):
        """
        Set up the parameters by asking for user inputs
        :return:
        """
        print('Dimensions Gabor Martrix: ' , self.Gabor_matrix.shape)
        start_time = input('Start time [in points]?:')
        if start_time == '':
            start_time = 5000
        else:
            start_time = int(start_time)
        end_time = input('End time [in points]?:')
        if end_time == '':
            end_time = 10000
        else:
            end_time = int(end_time)
        prop_time = input('Time for propagation?:')
        if prop_time == '':
            prop_time = 252
        else:
            prop_time = int(prop_time)
        px_size = input('Pixel size:?')
        if px_size == '':
            px_size = 4
        else:
            px_size = int(px_size)

        start_x = input('Where to start in x?:')
        if start_x == '':
            start_x = 0
        else:
            start_x = int(start_x)
        end_x = input('Where to end in x?:')
        if end_x == '':
            end_x = 1014
        else:
            end_x = int(end_x)
        self.initParameters(start_time, end_time,prop_time,px_size, start_x, end_x)

    def initParameters(self, start_time, end_time, prop_time, px_size, start_x, end_x):
        '''
        Initialization of the parameters
        :param start_time:
        :param end_time:
        :param prop_time:
        :param px_size:
        :param start_x:
        :param end_x:
        :return:
        '''
        self.start_time = start_time
        self.end_time = end_time
        self.prop_time = prop_time
        self.px_size = px_size
        self.start_x = start_x
        self.end_x = end_x

    def openCSV(self):
        ''' Open our cross correlation map
        Return: A numpy array of data
        '''
        print('Opening dataset: ' + self.path)
        list_col = []
        df = pd.read_csv(path)
        df_array = np.array(df)

        for column in df.columns:
            if 'Unnamed' in column:
                list_col.append(column[8:])

        first_col = int(list_col[0]) + 2
        last_col = int(list_col[1])
        self.Gabor_matrix = df_array[:, first_col:last_col]
        self.Aspect = self.Gabor_matrix.shape[1] / self.Gabor_matrix.shape[0]
        self.gabor_matrix = self.Gabor_matrix[1000:6000, 0:1018]
        self.aspect = self.gabor_matrix.shape[1] / self.gabor_matrix.shape[0]
        plt.imshow(self.gabor_matrix, aspect = self.aspect, cmap ='turbo')
        plt.show()


        if self.debug:
            self.displayImage('Gabor')

    def segmentation(self, mask_value, kernel_medianFilter = 25):
        '''
        Perform filtering and segmentation (threshold) of the Gabor map
        :param mask_value: threshold value
        :param kernel_medianFilter:
        :return:
        '''
        print('Segmenting image')
        self.gabor_matrix = self.Gabor_matrix[self.start_time:self.end_time, self.start_x:self.end_x]
        self.aspect = self.gabor_matrix.shape[1] / self.gabor_matrix.shape[0]
        self.mask_image = np.zeros(self.gabor_matrix.shape, dtype=np.uint8)
        filter_matrix = scipy.ndimage.median_filter(self.gabor_matrix, kernel_medianFilter)
        self.mask_image[filter_matrix < mask_value] = 1

        if self.debug:
            self.displayImage('Segmentation')

    def objectLabelling(self, min_size, max_size):
        '''
        Give a number to unconnected objects
        :param min_size:
        :param max_size:
        :return:
        '''
        print('Filtering Blob Sizes ')
        nb_blobs, image, stats, _ = cv2.connectedComponentsWithStats(self.mask_image)
        blob_sizes = stats[:, -1]

        self.blob_image = np.zeros(self.gabor_matrix.shape)
        list_blobs = []

        for blob in range(1,nb_blobs):
            if blob_sizes[blob] > min_size and blob_sizes[blob] < max_size:
                self.blob_image[image == blob] = blob
                list_blobs.append(blob)

        self.nb_blobs = len(list_blobs)
        self.stack_blobs = np.zeros((self.nb_blobs,self.blob_image.shape[0], self.blob_image.shape[1]))

        for i, blob in enumerate(list_blobs):
            im_to_process = np.zeros(self.gabor_matrix.shape, dtype=np.uint8)
            im_to_process[self.blob_image == blob] = 1
            self.stack_blobs[i] = im_to_process

        if debug:
            self.displayImage('Blob')

    def removeBlob(self):
        self.list2Remove = input('List of blobs to remove ? (speration , )' )
        if self.list2Remove == 'None':
            pass
        else:
            self.list2Remove= self.list2Remove.split(',')
            list_element = []
            for i, blob2r in enumerate(self.list2Remove):
                blobNum = int(blob2r)
                coord = np.where(self.blob_image == blobNum)
                x = coord[0][0]
                y = coord[1][0]

                for j, slice in enumerate(self.stack_blobs):
                    if slice[x,y] == 1:
                        list_element.append(j)

            self.stack_blobs = np.delete(self.stack_blobs,list_element,axis=0)
            self.nb_blobs -= len(list_element)

        #self.blob_image = np.zeros(self.gabor_matrix.shape)
        #for blob in range(1, nb_blobs):
            #self.blob_image[image == blob] = blob


        if debug:
            self.displayImage('Stack_Blob')

    def skeletonize(self, remove_small_obj = 300, fill_holes = 300):
        """
        Create skeleton for each blob and filter the branches
        :param remove_small_obj:
        :param fill_holes:
        :return:
        """
        self.stack_skeleton = np.zeros((self.nb_blobs, self.blob_image.shape[0], self.blob_image.shape[1]))

        for i, single_blob in enumerate(self.stack_blobs):
            print('Generating skeleton ', round(i/(self.nb_blobs-1) * 100), ' %')
            single_blob = binary_opening(binary_closing(single_blob, disk(2)), disk(2))
            single_blob = remove_small_objects(single_blob, remove_small_obj)
            single_blob = remove_small_holes(single_blob, fill_holes)
            skeleton = thin(single_blob)
            self.stack_skeleton[i] = skeleton

        #if debug:
            #self.displayImage('Skeleton')

    def filterSkeleton(self):
        '''
        Filtering the skeleton to find central path
        :return:
        '''
        print('Filtering Skeleton')
        for j, skeleton in enumerate(self.stack_skeleton):
            print('Filtering Skeleton: ' + str(j))
            graph_class = csr.Skeleton(skeleton)

            list_coord_branches = []
            for i in range(graph_class.n_paths):
                list_coord_branches.append(graph_class.path_coordinates(i))
            connect = csr.summarize(graph_class)
            connect.insert(0, "id", range(len(connect)), True)

            list_coord = []

            list_coord_to_save = []
            print(connect.to_string())
            for idbr in list(connect[connect['branch-type'] == 0]['id']):
                list_coord_to_save.append(list_coord_branches[int(idbr)])
            connect = connect[connect['branch-type'] != 0]

            index = 0
            if len(connect) != 0:
                while (1):
                    if index == 0:
                        first_branch = connect[connect['branch-type'] == 1]
                        maxLength = first_branch['euclidean-distance'].max()
                        first_branch = first_branch[first_branch['euclidean-distance'] == maxLength]

                        connect = connect[connect['euclidean-distance'] != maxLength]

                        fb_dst = first_branch['node-id-src'].values
                        fb_src = first_branch['node-id-dst'].values

                        if not (fb_dst in connect['node-id-src'].values or fb_dst in connect['node-id-dst'].values):
                            flag_start = 'dst'
                        elif not (fb_src in connect['node-id-src'].values or fb_src in connect['node-id-dst'].values):
                            flag_start = 'src'

                        fb_x0 = np.array(first_branch['image-coord-dst-0'])
                        fb_y0 = np.array(first_branch['image-coord-dst-1'])
                        fb_x1 = np.array(first_branch['image-coord-src-0'])
                        fb_y1 = np.array(first_branch['image-coord-src-1'])


                        if flag_start == 'dst':
                            first_angle = -1 * np.degrees(np.arctan2((fb_x0 - fb_x1), (fb_y0 - fb_y1)))
                            list_coord.append([fb_x1, fb_y1, fb_x0, fb_y0])
                            next_node = int(first_branch['node-id-dst'])

                        elif flag_start == 'src':
                            first_angle = -1 * np.degrees(np.arctan2((fb_x1 - fb_x0), (fb_y1 - fb_y0)))
                            list_coord.append([fb_x0, fb_y0, fb_x1, fb_y1])
                            next_node = int(first_branch['node-id-src'])

                        print('First Branch Next Node ', next_node)
                        print('First Branch Src ', fb_src)
                        print('First Branch Dst ', fb_dst)
                    else:
                        sub_branchs = connect[connect['node-id-dst'] == int(next_node)]
                        current_node = next_node

                        if len(sub_branchs) == 2:
                            sub_branch1 = sub_branchs.iloc[[0]]
                            sub_branch2 = sub_branchs.iloc[[1]]
                            connect = connect[connect['node-id-dst'] != int(next_node)]
                        elif len(sub_branchs) == 1:
                            sub_branch1 = sub_branchs
                            connect = connect[connect['node-id-dst'] != int(next_node)]

                        sub_branchs = connect[connect['node-id-src'] == int(next_node)]

                        if len(sub_branchs) == 2:
                            sub_branch1 = sub_branchs.iloc[[0]]
                            sub_branch2 = sub_branchs.iloc[[1]]
                            connect = connect[connect['node-id-src'] != int(next_node)]
                        elif len(sub_branchs) == 1:
                            sub_branch2 = sub_branchs
                            connect = connect[connect['node-id-src'] != int(next_node)]

                        angle1, next_node1, coord1, brType1 = self.computeAngle(sub_branch1, next_node)
                        angle2, next_node2, coord2, brType2 = self.computeAngle(sub_branch2, next_node)

                        error1 = abs(angle1 - first_angle)
                        error2 = abs(angle2 - first_angle)

                        if brType1 == 1 and brType2 ==2:
                            first_branch = sub_branch2
                            next_node = next_node2
                            list_coord.append(coord2)
                            first_angle = angle2

                        elif brType1 == brType2:
                            if error1 < error2:
                                first_branch = sub_branch1
                                next_node = next_node1
                                list_coord.append(coord1)
                                first_angle = angle1
                            else:
                                first_branch = sub_branch2
                                next_node = next_node2
                                list_coord.append(coord2)
                                first_angle = angle2

                        elif brType1 ==2 and brType2 == 1:
                            first_branch = sub_branch1
                            next_node = next_node1
                            list_coord.append(coord1)
                            first_angle = angle1
                        print('-----')
                        print('Current Brach : ',current_node)
                        print('Next Branchs : ', next_node1,next_node2 )
                        print('Branch choosen : ', next_node)
                        print('Branchs Type : ', brType1, brType2)
                        print('Angles : ',angle1, angle2)
                        print('------')

                    list_coord_to_save.append(list_coord_branches[int(first_branch['id'])])
                    if int(first_branch['branch-type']) == 1 and index != 0:
                        break

                    index += 1

            skeletontmp = np.zeros((skeleton.shape[0],skeleton.shape[1]))
            for br in list_coord_to_save:
                for c in br:
                   skeletontmp[int(c[0]), int(c[1])] = 1

            self.stack_skeleton[j] = skeletontmp

        if self.debug:
            #self.displayImage('Skeleton')
            self.displayImage('All_Skeletons')

    def fitSkeleton(self, r2_threshold_x, r2_threshold_y, min_nr_points_x, min_nr_points_y, min_len_line = 10):
        self.final_data = []
        self.listSlope = []
        for i, skeleton in enumerate(self.stack_skeleton):
            print('Fitting Skeleton: ' + str(i))
            skeleton = np.array(skeleton, dtype=np.uint8)
            coord = np.where(skeleton == 1)

            x = coord[0]
            y = coord[1]
            coord = np.array(list(zip(x, y)))
            self.list_coord = []
            self.fitBlob(coord, r2_threshold_x, r2_threshold_y, min_nr_points_x, min_nr_points_y) #, []
            print(self.listSlope)
            for par in self.listSlope:
                if par[-1]:
                    slope = par[0]
                    intercept = par[2]
                    y_points = (par[4], par[5])
                    x_points = np.multiply(slope, y_points) + intercept
                    speed = slope

                else:
                    slope = par[1]
                    intercept = par[3]
                    x_points = (par[6], par[7])
                    y_points = np.multiply(slope, x_points) + intercept

                    if slope != 0:
                        speed = (1.0/slope)
                    else:
                        speed = 10

                distance = np.sqrt(((y_points[0] - y_points[1]) ** 2 + (x_points[0] - x_points[1]) ** 2))
                if distance > min_len_line:
                    self.final_data.append([speed,intercept,x_points, y_points])

    def displayImage(self, flagImageToDisplay, vmin = -200 , vmax = 400 ):
        '''
        Displays images during the analysis
        :param flagImageToDisplay:
        :param vmin:
        :param vmax:
        :return:
        '''
        if flagImageToDisplay == 'Gabor':
            plt.imshow(np.flip(self.Gabor_matrix), vmin = vmin, vmax = vmax, cmap = 'turbo', aspect= self.Aspect)
            plt.show()

        elif flagImageToDisplay == 'Segmentation':
            plt.imshow(np.flip(self.mask_image), aspect= self.aspect)
            plt.show()

        elif flagImageToDisplay == 'Blob':
            plt.imshow(np.flip(self.blob_image), aspect= self.aspect)
            plt.show()

        elif flagImageToDisplay == 'Skeleton':
            for i in range(self.nb_blobs):
                plt.imshow(np.flip(self.stack_skeleton), aspect=self.aspect)
                plt.show()

        elif flagImageToDisplay == 'Stack_Blob':
            for i in range(len(self.stack_blobs)):
                plt.imshow(self.stack_blobs[i], aspect=self.aspect)
            plt.show()


        elif flagImageToDisplay == 'All_Skeletons':
            full_skeleton = np.sum(self.stack_skeleton,axis=0)
            print(self.stack_skeleton.shape)
            print(full_skeleton.shape)
            plt.imshow(full_skeleton, aspect= self.aspect, cmap = 'turbo')
            plt.show()
            #for i in range(len(self.stack_skeleton)):
                #plt.imshow(self.stack_skelton[i], aspect=self.aspect)
            #plt.show()


    def fitBlob(self,coord,r2_threshold_x = 0.8, r2_threshold_y = 0.8, min_nr_points_x = 20, min_nr_points_y = 20): #,listSlope
        nb_points_x = self.gabor_matrix.shape[0]
        nb_points_y = self.gabor_matrix.shape[1]
        ratio = nb_points_y/nb_points_x
        x = [c[0] for c in coord]
        y = [c[1] for c in coord]
        self.list_coord.append([x,y])
        x0 = x[0]
        print(x0)
        x1 = x[-1]
        print(x1)

        y0 = np.min(y)
        print(y0)

        y1 = np.max(y)
        print(y1)

        xdiff = (x1-x0)/5
        ydiff = (y1-y0)
        print(xdiff,ydiff)

        if xdiff > ydiff:
            slopex, interceptx, r_valuex, p_valuex, std_errx = scipy.stats.linregress(x, y)
            slopey = np.nan
            intercepty = np.nan
            HorizontalFlag = True
            #yfity = np.multiply(slopey, x) + intercepty
            #errory = np.sum((yfity - y) ** 2) / len(x)
            r2 = r_valuex ** 2
            r2_threshold = r2_threshold_x
            min_nr_points = min_nr_points_x

        else:
            slopey, intercepty, r_valuey, p_valuey, std_erry = scipy.stats.linregress(y, x)
            slopex = np.nan
            interceptx = np.nan
            HorizontalFlag = False
            #yfitx = np.multiply(1.0 / slopex, x) - (1.0 * interceptx / slopex)
            #errorx = np.sum((yfitx - y) ** 2) / len(x)
            r2 = r_valuey** 2
            r2_threshold = r2_threshold_y
            min_nr_points = min_nr_points_y

        #if slopey == 0.0:
            #slope = 0.0
            #intercept = intercepty
            #r2 = 1
            #HorizontalFlag = False

        #elif slopex == 0.0:
            #slope = 10.0
            #intercept = x[0]
            #r2 = 1
            #HorizontalFlag = True
            #x0 = np.min(y)
            #x1 = np.max(y)

        #elif errory <= errorx:
            #slope = slopey
            #intercept = intercepty
            #r2 = r2y
            #HorizontalFlag = False
        #else:
            #slope = 1.0 / slopex
            #intercept = -1.0 * interceptx / slopex
            #r2 = r2x
            #HorizontalFlag = False
        print(r2)
        if (r2 < r2_threshold) and (len(x) > min_nr_points) :
            cluster = np.array(AgglomerativeClustering(2).fit(coord).labels_, dtype=bool)
            coord1 = coord[cluster == True]
            coord2 = coord[cluster == False]

            if (len(coord1) > min_nr_points) and (len(coord2) > min_nr_points):
                self.fitBlob(coord1,r2_threshold_x, r2_threshold_y , min_nr_points_x , min_nr_points_y ) #, listSlope
                self.fitBlob(coord2,r2_threshold_x , r2_threshold_y , min_nr_points_x, min_nr_points_y )

            else:
                self.listSlope.append([slopex, slopey, interceptx, intercepty, x0, x1, y0,y1,HorizontalFlag])
                print(self.listSlope)
                for coordi in self.list_coord:
                    xd = coordi[0]
                    yd = coordi[1]
                    plt.plot(xd, yd, '*')
                for par in self.listSlope:
                    if par[-1]:
                        x_points = (par[4], par[5])
                        y_points = np.multiply(par[0], x_points) + par[2]
                    else:
                        y_points = (par[6], par[7])
                        x_points = np.multiply(par[1], y_points) + par[3]

                    plt.plot(x_points,y_points)
                    print(x_points, y_points)

                #plt.show()

        else:
            self.listSlope.append([slopex,slopey, interceptx,intercepty, x0, x1,y0,y1, HorizontalFlag]) #listSlope
            print(self.listSlope)
            for coordi in self.list_coord:
                xd = coordi[0]
                yd = coordi[1]
                plt.plot(xd, yd, '*')
            for par in self.listSlope:
                if par[-1]:
                    x_points = (par[4], par[5])
                    y_points = np.multiply(par[0], x_points) + par[2]
                else:
                    y_points = (par[6], par[7])
                    x_points = np.multiply(par[1], y_points) + par[3]
                plt.plot(x_points, y_points)
                print(x_points, y_points)
            plt.show()



    def computeAngle(self,subBranch,currentNode):

        if int(subBranch['node-id-src']) == currentNode:
            next_node = int(subBranch['node-id-dst'])
            flag_start = 'dst'
        else:
            next_node = int(subBranch['node-id-src'])
            flag_start = 'src'

        fb_x0 = np.array(subBranch['image-coord-dst-0'])
        fb_y0 = np.array(subBranch['image-coord-dst-1'])
        fb_x1 = np.array(subBranch['image-coord-src-0'])
        fb_y1 = np.array(subBranch['image-coord-src-1'])
        branch_type = np.array(subBranch['branch-type'])

        if flag_start == 'dst':
            angle = -1 * np.degrees(np.arctan2((fb_x0 - fb_x1), (fb_y0 - fb_y1)))
            coord = [fb_x1, fb_y1, fb_x0, fb_y0]

        elif flag_start == 'src':
            angle = -1 * np.degrees(np.arctan2((fb_x1 - fb_x0), (fb_y1 - fb_y0)))
            coord = [fb_x0, fb_y0, fb_x1, fb_y1]

        return angle, next_node, coord,branch_type

########################################################################################################################

    def displayFinalData(self, vmin, vmax):

        self.x_count = self.gabor_matrix.shape[1]
        self.t_count = self.gabor_matrix.shape[0]
        self.X_count = self.Gabor_matrix.shape[1]
        self.T_count = self.Gabor_matrix.shape[0]
        Gabor_shape = self.Gabor_matrix.shape

        fig = plt.figure(figsize=(20, 7))

        h1 = [Size.Fixed(1.0), Size.Fixed(5.0)]
        v1 = [Size.Fixed(1.0), Size.Fixed(5.0)]
        divider1 = Divider(fig, (0, 0, 1, 1), h1, v1, aspect=False)

        ax1 = fig.add_axes(divider1.get_position(), axes_locator=divider1.new_locator(nx=1, ny=1))
        ax1.imshow(np.flip(self.Gabor_matrix, axis = 0), cmap = 'turbo', aspect=self.Aspect, vmin = vmin, vmax =vmax)
        ax1.add_patch(Rectangle((self.start_x, Gabor_shape[0] - self.end_time), self.end_x - self.start_x, self.end_time - self.start_time, linewidth=2, edgecolor='white', facecolor='none'))
        #ax1.add_patch(Rectangle((self.start_x, self.start_time), self.end_x - self.start_x, self.end_time - self.start_time, linewidth=2, edgecolor='#000000', facecolor='none'))
        ax1.set_xlabel('X cross-section (mm)', fontsize=14, fontname = "Century")
        ax1.set_title('Correlation Magnitude', fontname = "Century", fontsize=14)
        ax1.set_ylabel('Time (s)', fontsize=14, fontname = "Century")

        x_array = np.arange(0, Gabor_shape[1])  # the grid to which your data corresponds
        no_labels = 5  # how many labels to see on axis x
        step_x = int(Gabor_shape[1] / (no_labels))  # step between consecutive labels
        x_positions = np.arange(0, Gabor_shape[1], step_x)  # pixel count at label position
        x_labels = x_array[::step_x] / 10 ** 3  # labels you want to see
        ax1.set_xticks(x_positions, np.round(x_labels * self.px_size, 2),fontname = "Century", fontsize=14)

        time = np.arange(0, Gabor_shape[0])  # the grid to which your data corresponds
        time = time[::-1]
        no_labels = 5  # how many labels to see on axis x
        step_time = int(Gabor_shape[0] / (no_labels))  # step between consecutive labels
        time_positions = np.arange(0, Gabor_shape[0], step_time)  # pixel count at label position
        time_labels = time[::step_time] / 5000.0  # labels you want to see
        ax1.set_yticks(time_positions, np.round(time_labels + self.prop_time, 2), fontname = "Century", fontsize=13)

        ax1.set_ylim(self.T_count,0)
        ax1.set_xlim(0,self.X_count)

#########################################

        h4 = [Size.Fixed(1.9), Size.Fixed(10.0)]
        v4 = [Size.Fixed(1.0), Size.Fixed(5.0)]
        divider4 = Divider(fig, (0, 0, 1, 1), h4, v4, aspect=False)
        ax4 = fig.add_axes(divider4.get_position(),axes_locator=divider4.new_locator(nx=1, ny=1))
        ax4.imshow(self.gabor_matrix, cmap = 'turbo', aspect=self.aspect, vmin = vmin, vmax = vmax)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = ax4.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='turbo'),ax=ax4, shrink=0.71, fraction =0.40)
        cbt = cbar.get_ticks().tolist()
        cbar.set_ticklabels(cbt, fontname = "Century", fontsize = 14)

        ax4.set_xlabel('X cross-section (mm)', fontsize=14, fontname = "Century")
        ax4.set_title('ROI Correlation Magnitude', fontname = "Century", fontsize=14)
        ax4.set_ylabel('Time (s)', fontsize=14, fontname = "Century")


        x_array = np.arange(self.start_x, self.end_x)  # the grid to which your data corresponds
        no_labels = 5  # how many labels to see on axis x
        step_x = int(self.x_count / (no_labels))  # step between consecutive labels
        x_positions = np.arange(0, self.x_count, step_x)  # pixel count at label position
        x_labels = x_array[::step_x] / 10 ** 3  # labels you want to see
        ax4.set_xticks(x_positions, np.round(x_labels * self.px_size, 2), fontname = "Century", fontsize=14)

        time = np.arange(self.start_time, self.end_time)  # the grid to which your data corresponds
        #time = time[::-1]
        no_labels = 5  # how many labels to see on axis x
        step_time = int(self.t_count / (no_labels))  # step between consecutive labels
        time_positions = np.arange(0, self.t_count, step_time)  # pixel count at label position
        time_labels = time[::step_time] / 5000.0  # labels you want to see
        #time_labels = time_labels[::-1]
        ax4.set_yticks(time_positions, np.round(time_labels + self.prop_time, 2), fontname = "Century", fontsize=14)



        for data in self.final_data:
            ax4.plot(data[2], data[3], color='white', linewidth=2)

        ax4.invert_yaxis()
        #ax4.flip()

        ax4.set_ylim(0,self.t_count)
        ax4.set_xlim(0, self.x_count)


        h3 = [Size.Fixed(7.2), Size.Fixed(18.0)]
        v3 = [Size.Fixed(1.0), Size.Fixed(5.0)]
        divider3 = Divider(fig, (0, 0, 1, 1), h3, v3, aspect=False)
        ax3 = fig.add_axes(divider3.get_position(), axes_locator=divider3.new_locator(nx=1, ny=1))



        tmin = 10000
        tmax = -1
        self.speed_list = []
        self.time_list = []
        for i, v in enumerate(self.final_data):
            slope = (v[0] * 5000 * 20) / (10 ** 6)
            if np.abs(slope) < 50:
                self.speed_list.append(slope)
                x = self.prop_time + self.start_time / 5000 + v[3][0] / 5000
                self.time_list.append(x)
                if tmin >  (v[3][0] / 5000):
                    tmin =  (v[3][0] / 5000)
                if tmax <  (v[3][0] / 5000):
                    tmax = (v[3][0] / 5000)

                ax3.plot(x, np.abs(slope), color='black', marker='o')

        tmin = tmin * 0.95
        tmax = tmax * 1.05
        delta = tmax-tmin
        ax3.set_aspect(1. / ax3.get_data_ratio())
        ticks_x = np.round((np.arange(0,8)*delta/8.0) + tmin + (self.prop_time + self.start_time / 5000),2)
        ax3.set_xticks(ticks_x, fontname = "Century")
        plot_font = font_manager.FontProperties(family='Century', style='normal', size= 13, weight='normal', stretch='normal')

        for label in ax3.get_xticklabels():
            label.set_fontproperties(plot_font)

        for label in ax3.get_yticklabels():
            label.set_fontproperties(plot_font)

        ax3.ticklabel_format(useOffset=False)


        ax3.set_xlabel('Time (s)', fontsize=14, fontname = "Century")
        ax3.set_ylabel('Horizontal speed (m/s)', fontsize=14, fontname = "Century")
        ax3.set_title('Reaction Front Delamination Speed', fontsize=14, fontname = "Century")
        ax3.grid()

        fig.suptitle(self.title, fontsize=16, fontname = "Century")
        plt.show()

        self.number_lines = i

    def saveParameters(self, path):
        path = '\\'.join(path.split('\\')[:-1])
        print(path)
        string_list = [('Time:' + str(self.start_time) + '-' + str(self.end_time)), ('X:' + str(self.start_x) + '-' + str(self.end_x)),
                       ('Minimum blob size:' + str(minimum_blob_size)), ('Maximum blob size:' + str(maximum_blob_size)),
                       ('Vmin:' + str(vmin)), ('Vmax:' + str(vmax)),
                       ('Segmentation Threshold:' + str(segmentation_threshold)), ('r2 threshold_x:' + str(r2_threshold_x)), ('r2 threshold_y:' + str(r2_threshold_y)), ('min_fit_line_nb:' + str(min_fit_line_nb_x)), ('min_fit_line_nb:' + str(min_fit_line_nb_y)), ('min_length_line:' +str(min_length_line)), ('Speed:' + str(self.speed_list)), ('Time:' + str(self.time_list)), ('List to remove:' +str(self.list2Remove))]

        filename = path+ 'config.txt'
        f = open(filename, "w")
        for string in string_list:
            f.write(string + "\n")
        print('Parameters Saved')

if __name__ == '__main__':
    path = 'W:\\Data\\data_processing_mi1354\\Gabor filtering\\M50_PT_Exp3\\2\\Cross_correlations.csv'
    debug = True
    # Image Analysis
    vmin = -500
    vmax = 900
    segmentation_threshold = 0
    minimum_blob_size = 20000
    maximum_blob_size = 200000000000
    r2_threshold_x = 0.1
    r2_threshold_y = 0.1
    min_fit_line_nb_x = 50
    min_fit_line_nb_y = 50
    min_length_line = 0

    analyser = GaborLineDetection(path, debug)
    analyser.openCSV()
    analyser.setupParameters()
    analyser.segmentation(segmentation_threshold)
    analyser.objectLabelling(minimum_blob_size, maximum_blob_size)
    analyser.removeBlob()
    analyser.skeletonize()
    analyser.filterSkeleton()
    analyser.fitSkeleton(r2_threshold_y,r2_threshold_x,min_fit_line_nb_y, min_fit_line_nb_x, min_length_line)
    analyser.displayFinalData(vmin, vmax)
    analyser.saveParameters(path)
