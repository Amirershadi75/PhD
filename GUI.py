#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 22:59:19 2022

@author: amir
"""

import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import loaddataset as ld
from tkinter import Tk
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
import pre_data as pre_data
import regres as regres
import visual as visual
import matclassnrgui as mcnrgui
import seaborn as sns

# from pandastable import Table, TableModel

## Load dataset
Output,material_class,test_type=ld.load_dataset()


##Tkinter GUI


# from pandastable import Table, TableModel

# GUI Window

############
win=Tk()
screen_width=1440
screen_height=900
title = win.title("Leaching_Database")
win.geometry(f'{screen_width}x{screen_height}')
# win.configure(background='white')
##Test type and Material class frame
frame_A = LabelFrame(win,text='Leaching_data',font = "Arial 14 bold")
frame_A.grid(row=0,column=0,sticky='W')

Label(frame_A,text='Material',font = "Arial 14  ").grid(row=0,sticky='W', padx=5, pady=20)
Label(frame_A,text='Test Type',font = "Arial 14  ").grid(row=1,sticky='W', padx=5, pady=10)
  
   
LS_option=['higher than','lower than', 'equal','higher and equal','lower and equal','no filter']
frame_B = LabelFrame(win,text='LS Filter',font = "Arial 14 bold")
frame_B.place(x=2,y=130)
Label(frame_B,text='LS',font = "Arial 14  ").grid(row=0,sticky='W', padx=5, pady=10) 
chosen_ls = StringVar(frame_B)
ls_option = OptionMenu(frame_B, chosen_ls, *sorted(LS_option))
ls_option.grid(row=0, column=1, sticky='W', padx=1, pady=2)
ls_value = Entry(master=frame_B, width=5)
ls_value.grid(row=0,column=2,sticky='W', padx=2, pady=2)
save_value = Button(master =frame_B, command =lambda:run(),text ='Apply') 
save_value .grid(row=1, column=0, sticky='W', padx=1, pady=3) 


# save_value = Button(master =frame_A, command =lambda:run(),text ='Run') 
# save_value .grid(row=2, column=0, sticky='W', padx=5, pady=3) 

chosen_mat=list(np.zeros(len(material_class)))
chosen_type=list(np.zeros(len(test_type)))


col=Output.columns.values   
frame_C = LabelFrame(win,text='Plot Filter',font = "Arial 14 bold")
frame_C.place(x=2,y=250)
frame_D = LabelFrame(win,text='Machine learning frame',font = "Arial 14 bold")
frame_D.place(x=295,y=130)

Label(frame_C,text='X_axis',font = "Arial 14  ").grid(row=0,sticky='W', padx=5, pady=10) 
Label(frame_C,text='Y_axis',font = "Arial 14  ").grid(row=1,sticky='W', padx=5, pady=10) 
chosen_x = StringVar(frame_C)
chosen_y = StringVar(frame_C)
x_axis = OptionMenu(frame_C, chosen_x, *sorted(col))
x_axis.grid(row=0, column=1, sticky='W', padx=1, pady=2)
y_axis = OptionMenu(frame_C, chosen_y, *sorted(col))
y_axis.grid(row=1, column=1, sticky='W', padx=1, pady=2)
save_value = Button(master =frame_C, command =lambda:plot(),text ='Plot') 
save_value .grid(row=2, column=0, sticky='W', padx=5, pady=3) 

machine_learning_run = Button(master =frame_D, command =lambda:machine_run(cvv,n_unique,n_estimatorss),text ='material prediction') 
machine_learning_run .grid(row=4, column=0, sticky='W', padx=5, pady=3) 


### number of unique sample for test data
Label(frame_D,text='Number of unique sample for test',font = "Arial 14  ").grid(row=0,sticky='W', padx=5, pady=10) 
n_unique = Entry(master=frame_D, width=5)
n_unique.grid(row=0,column=1,sticky='W', padx=2, pady=2)
n_unique.insert(END, 70)
### Cross validation
Label(frame_D,text='Number of cross validation fold',font = "Arial 14  ").grid(row=1,sticky='W', padx=5, pady=10) 
cvv = Entry(master=frame_D, width=5)
cvv.grid(row=1,column=1,sticky='W', padx=2, pady=2)
cvv.insert(END, 10)
### Number of trees in the forest
Label(frame_D,text='n_estimators',font = "Arial 14  ").grid(row=2,sticky='W', padx=5, pady=10) 
n_estimatorss = Entry(master=frame_D, width=5)
n_estimatorss.grid(row=2,column=1,sticky='W', padx=2, pady=2)
n_estimatorss.insert(END, 10)


##########


def run():      
    global chosen_mat
    global chosen_type
    global Output_back
    global yy
    global xx
    global result_data
    result_data=[]
    result_data=pd.DataFrame(result_data)
    Output_back=Output
    chosen_mat=list(np.zeros(len(material_class)))
    chosen_type=list(np.zeros(len(test_type)))
    unit=Output.loc[Output['Column_1'] == 'Unit']       
    result_data=pd.concat([result_data,unit])
    try:
        ls_opt=str(chosen_ls.get())
    except:
        pass
    try:
        ls_val=float(ls_value.get())
    except:
        pass
    
    try:
        xx=str(chosen_x.get())
        #print(xx)
    except:
        pass
    try:
        yy=str(chosen_y.get())
        #print(yy)
    except:
        pass
##Results table frame     
    for i in range (len(material_class)):  
        
        j=i
        i=str(i+1)
        if eval('mat_class_'+i+'.get()')==1:
            chosen_mat[j]=material_class[j]
    for i in range (len(test_type)):  
        j=i
        i=str(i+1)
        if eval('testtype_'+i+'.get()')==1:
            chosen_type[j]=test_type[j]        
    for i in range (len(material_class)):  
        
        chosen_mat=pd.DataFrame(chosen_mat)
        
        if chosen_mat[0][i]!= 0:
            
            Output_back=Output.loc[Output['MaterialClass'] == chosen_mat[0][i] ]
            result_data=pd.concat([result_data,Output_back])
            un=Output.loc[Output['Column_1']== 'Unit' ]
            if    ls_opt=='higher than':
                result_data=result_data.loc[result_data['LS'] > ls_val ]
                result_data=pd.concat([un,result_data])
            if    ls_opt=='lower than':
                result_data=result_data.loc[result_data['LS'] < ls_val ]  
                result_data=pd.concat([un,result_data])
            if    ls_opt=='higher and equal':
                result_data=result_data.loc[result_data['LS'] >= ls_val ] 
                result_data=pd.concat([un,result_data])
            if    ls_opt=='lower and equal':
                result_data=result_data.loc[result_data['LS'] <= ls_val ]
                result_data=pd.concat([un,result_data])
            if    ls_opt=='equal':
                result_data=result_data.loc[result_data['LS'] == ls_val ] 
                result_data=pd.concat([un,result_data])
    chosen_type=pd.DataFrame(chosen_type)           
    count=(chosen_type!= 0).sum()        
    count=count[0]
    count_mat=(chosen_mat!= 0).sum()        
    count_mat=count_mat[0]
    if count!=0 and count_mat!=0 :        
        filter_back=result_data   
        back=[]
        back=pd.DataFrame(back)
        back=pd.concat([back,unit])
        Output_back=result_data   
        result_data=back
        for i in range (len(test_type)):       
            chosen_type=pd.DataFrame(chosen_type)   
           
    
            if chosen_type[0][i]!= 0:
                Output_back=filter_back.loc[filter_back['Test_type'] == chosen_type[0][i] ]
                result_data=pd.concat([result_data,Output_back])
                un=Output.loc[Output['Column_1']== 'Unit' ]
                if    ls_opt=='higher than':
                    result_data=result_data.loc[result_data['LS'] > ls_val ]
                    result_data=pd.concat([un,result_data])
                if    ls_opt=='lower than':
                    result_data=result_data.loc[result_data['LS'] < ls_val ]   
                    result_data=pd.concat([un,result_data])
                if    ls_opt=='higher and equal':
                    result_data=result_data.loc[result_data['LS'] >= ls_val ]
                    result_data=pd.concat([un,result_data])
                if    ls_opt=='lower and equal':
                    result_data=result_data.loc[result_data['LS'] <= ls_val ] 
                    result_data=pd.concat([un,result_data])
                if    ls_opt=='equal':
                    result_data=result_data.loc[result_data['LS'] == ls_val ]     
                    result_data=pd.concat([un,result_data])
    if count!=0 and count_mat==0 :            

        for i in range (len(test_type)): 
            chosen_type=pd.DataFrame(chosen_type)   
            if chosen_type[0][i]!= 0:
                Output_back=Output.loc[Output['Test_type'] == chosen_type[0][i] ]
                result_data=pd.concat([result_data,Output_back])
                un=Output.loc[Output['Column_1']== 'Unit' ]
                if    ls_opt=='higher than':
                    result_data=result_data.loc[result_data['LS'] > ls_val ]
                    result_data=pd.concat([un,result_data])
                if    ls_opt=='lower than':
                    result_data=result_data.loc[result_data['LS'] < ls_val ]   
                    result_data=pd.concat([un,result_data])
                if    ls_opt=='higher and equal':
                    result_data=result_data.loc[result_data['LS'] >= ls_val ]  
                    result_data=pd.concat([un,result_data])
                if    ls_opt=='lower and equal':
                    result_data=result_data.loc[result_data['LS'] <= ls_val ] 
                    result_data=pd.concat([un,result_data])
                if    ls_opt=='equal':
                    result_data=result_data.loc[result_data['LS'] == ls_val ] 
                    result_data=pd.concat([un,result_data])
    if count==0 and count_mat==0 : 
            result_data=Output      
            un=Output.loc[Output['Column_1']== 'Unit' ]
            if    ls_opt=='no filter':
                result_data=result_data
            if    ls_opt=='higher than':
                result_data=Output.loc[Output['LS'] > ls_val ]
                result_data=pd.concat([un,result_data])
            if    ls_opt=='lower than':
                result_data=Output.loc[Output['LS'] < ls_val ]   
                result_data=pd.concat([un,result_data])
            if    ls_opt=='higher and equal':
                result_data=Output.loc[Output['LS'] >= ls_val ]  
                result_data=pd.concat([un,result_data])
            if    ls_opt=='lower and equal':
                result_data=Output.loc[Output['LS'] <= ls_val ]  
                result_data=pd.concat([un,result_data])
            if    ls_opt=='equal':
                result_data=Output.loc[Output['LS'] == ls_val ]
                result_data=pd.concat([un,result_data])
               
    res= LabelFrame(win, text='Result', font="Arial 12 bold italic")
    res.place(relx=0.5, rely=0.85,height=200,width=1400, anchor=CENTER)    
    
    tv1=ttk.Treeview(res)
    tv1.place(relheight=1,relwidth=1)
    treescrolly=Scrollbar(res,orient='vertical',command=tv1.yview)
    treescrollx=Scrollbar(res,orient='horizontal',command=tv1.xview)
    tv1.configure(xscrollcommand=treescrollx.set,yscrollcommand=treescrolly.set)
    treescrollx.pack(side='bottom',fill='x')
    treescrolly.pack(side='right',fill='y')
    tv1['column']=list(result_data.columns)
    tv1['show']='headings'
    for column in tv1['columns']:
        tv1.heading(column,text=column)
    
    Output_rows=result_data.to_numpy().tolist()
    for row in Output_rows:
        tv1.insert('','end',values=row)
mat_class_1 = IntVar() 
mat_class_2 = IntVar()    
mat_class_3 = IntVar()    
mat_class_4 = IntVar()    
mat_class_5 = IntVar()    
mat_class_6 = IntVar()    
mat_class_7 = IntVar()    
mat_class_8 = IntVar()       
mat_class_9 = IntVar()    
mat_class_10 = IntVar()   
testtype_1 = IntVar()    
testtype_2 = IntVar()   
testtype_3 = IntVar()   
testtype_4 = IntVar()   
testtype_5 = IntVar()   
testtype_6 = IntVar()      
testtype_7 = IntVar()        
for i in range (len(material_class)):  
    j=i
    i=str(i+1)
    material_class_option = Checkbutton(frame_A,text=material_class[j],variable=eval('mat_class_'+i), onvalue=1, offvalue=0, command=run)
    material_class_option.grid(row=0, column=j+1, sticky=W+E, padx=5, pady=2)

for i in range (len(test_type)):   
    j=i
    i=str(i+1)
    test_type_option = Checkbutton(frame_A,text=test_type[j],variable=eval('testtype_'+i), onvalue=1, offvalue=0, command=run)
    test_type_option.grid(row=1, column=j+1, sticky=W+E, padx=5, pady=2)

          
def plot():    
    run()
    #print(result_data[1:])  
    # plsql=Tk()
    # screen_width=1440
    # screen_height=900 
    # plsql.geometry(f'{screen_width}x{screen_height}')
    # title = plsql.title('FIGURE ' +xx+' VS '+yy)
    # plotplot = LabelFrame(plsql)
    # plotplot.place(x=1,y=8,width='1275',height='800')
    # aa=result_data['Probe'].unique()
    # aa=list(aa)
    # for widget in plotplot.winfo_children():
    #                 widget.destroy()   
    #f = plt.figure(figsize=(16, 10))
    #ax = f.add_subplot(111)
    # for i in range (len(aa)):
    #     X_val=np.array(result_data[xx].loc[result_data['Probe']==aa[i]  ], dtype=np.float)
    #     Y_val= np.array(result_data[yy].loc[result_data['Probe']==aa[i]  ], dtype=np.float)
    import plotly.express as px
    #     try:    
                
    
                
    #             #plt.plot(X_val,Y_val,'o')
    #             plt.plot(X_val,Y_val,'-o')
    #             # px.scatter(X_val,Y_val)
    
    #             # plt.xlim(xmin=-0.2)
    #             # plt.hold(True)
    
                
       
    #     except:
    #         pass
        
    # canvas = FigureCanvasTkAgg(f, master =plotplot)
    # canvas.draw()

    # canvas.get_tk_widget().pack()

    # # toolbarFrame = Frame(master=plotplot)
    # # toolbarFrame.pack()    
    # toolbar = NavigationToolbar2Tk(canvas, plotplot)

    # canvas._tkcanvas.pack()        
    # # plt.grid()
    # ax.legend(aa,ncol=2,loc='best',prop={'size': 12}) 
    # from matplotlib.ticker import FormatStrFormatter
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # plt.xticks(fontsize=8)
    # plt.yticks(fontsize=8)

    # try:
    #     plt.xlabel(xx+' ['+result_data[xx][0]+']',fontsize=12)
    # except:    
    #     plt.xlabel(xx,fontsize=12)
    # try:    
    #     plt.ylabel(yy+' ['+result_data[yy][0]+']',fontsize=12)
    # except:
    #     plt.ylabel(yy, fontsize=12)
    # #plt.show()
    # if xx=='pH' or xx=='LS' or xx=='ID' or xx=='Probe' or xx=='Test_type' :
    #     fig =px.line(result_data[1:],x=xx, y=yy,color="Probe", hover_data = ["Probe", "Test_type"], markers=True,symbol=result_data['Test_type'][1:],
    #         labels={ # replaces default labels by column name
    #                     str(yy): str(yy)+' ['+str(result_data[yy][0])+']'
    #         })    
    # if yy=='pH' or yy=='LS' or yy=='ID' or yy=='Probe' or yy=='Test_type':
    #     fig =px.line(result_data[1:],x=xx, y=yy,color="Probe", hover_data = ["Probe", "Test_type"], markers=True,symbol=result_data['Test_type'][1:],
    #         labels={ # replaces default labels by column name
    #                     str(xx): str(xx)+' ['+str(result_data[xx][0])+']'
    #                 }) 
    # print(str(result_data[xx][0]))
    # print(str(result_data[yy][0]))
    # print(type(str(result_data[xx][0])))
    # print(type(str(result_data[yy][0])))
    
    try:
        if str(result_data[yy][0])=='nan' or str(result_data[yy][0])=='None' and str(result_data[xx][0])!='nan' and str(result_data[xx][0])!='None':
            fig =px.line(result_data[1:],x=xx, y=yy,color="Probe", hover_data = ["Probe"], markers=True,symbol=result_data['Test_type'][1:],
                labels={ # replaces default labels by column name
                            str(xx): str(xx)+' ['+str(result_data[xx][0])+']','symbol':'Test type'
                        }) 
        if str(result_data[xx][0])=='nan' or str(result_data[xx][0])=='None' and str(result_data[yy][0])!='nan' and str(result_data[yy][0])!='None':
            fig =px.line(result_data[1:],x=xx, y=yy,color="Probe", hover_data = ["Probe"], markers=True,symbol=result_data['Test_type'][1:],
                labels={ # replaces default labels by column name
                            str(yy): str(yy)+' ['+str(result_data[yy][0])+']','symbol':'Test type'
                        })                 
        if str(result_data[yy][0])=='nan' and str(result_data[xx][0])=='nan':
            fig =px.line(result_data[1:],x=xx, y=yy,color="Probe", hover_data = ["Probe"], markers=True,symbol=result_data['Test_type'][1:],
                labels={ # replaces default labels by column name
                            'symbol':'Test type'
                        }) 
        if str(result_data[yy][0])=='None' and str(result_data[xx][0])=='None':
            fig =px.line(result_data[1:],x=xx, y=yy,color="Probe", hover_data = ["Probe"], markers=True,symbol=result_data['Test_type'][1:],
                labels={ # replaces default labels by column name
                            'symbol':'Test type'
                        })                  
        if str(result_data[yy][0])=='nan' and str(result_data[xx][0])=='None':
            fig =px.line(result_data[1:],x=xx, y=yy,color="Probe", hover_data = ["Probe"], markers=True,symbol=result_data['Test_type'][1:],
                labels={ # replaces default labels by column name
                            'symbol':'Test type'
                        })                  
        if str(result_data[xx][0])=='nan' and str(result_data[yy][0])=='None':
            fig =px.line(result_data[1:],x=xx, y=yy,color="Probe", hover_data = ["Probe"], markers=True,symbol=result_data['Test_type'][1:],
                labels={ # replaces default labels by column name
                            'symbol':'Test type'
                        })   
        if str(result_data[yy][0])!='nan' and str(result_data[yy][0])!='None' and str(result_data[xx][0])!='nan' and str(result_data[xx][0])!='None':
            fig =px.line(result_data[1:],x=xx, y=yy,color="Probe", hover_data = ["Probe"], markers=True,symbol=result_data['Test_type'][1:],
                labels={ # replaces default labels by column name
                            str(xx): str(xx)+' ['+str(result_data[xx][0])+']',str(yy): str(yy)+' ['+str(result_data[yy][0])+']','symbol':'Test type'
                        })       
    except:       
        fig =px.line(result_data[1:],x=xx, y=yy,color="Probe", hover_data = ["Probe"], markers=True,symbol=result_data['Test_type'][1:],
                labels={ # replaces default labels by column name
                            'symbol':'Test type'
                        })                                         
            

    
    fig.update_layout(autotypenumbers='convert types')
    fig.write_html("/Users/amir/Desktop/PhD/file.html")
    import webbrowser

    url = "file:///Users/amir/Desktop/PhD/file.html"

    webbrowser.open(url, new=0, autoraise=True)
    mainloop()

def machine_run(cvv,n_unique,n_estimatorss):
    GUI=True
    n=int(n_unique.get())
    cv=int(cvv.get())
    n_estimators=int(n_estimatorss.get())
    ########## Load dataset

    Output,material_class,test_type=ld.load_dataset()

    ########## Data preprocessing

    Output,machine_data,modified_data,Raw_data=pre_data.machinelearning_data(Output)

    ########## Material  classifier no repetition

    X,y,X_train,X_test,y_train,y_test,prediction_ExtraTrees,prediction_RandomForest,r2_test_ExtraTrees,r2_test_RandomForest,r2_train_ExtraTrees,\
         r2_train_RandomForest,X_new_ExtraTreesClassifier,X_new_RandomForestClassifier,feature_name_ExtraTrees,feature_name_RandomForest,srf,sxt,prf,pxt,cvrf,cvxt,corext,corrf,correlation_nomodel,feature_name_X_selected\
             =mcnrgui.material_class_norepeat_GUI(machine_data,GUI,n,cv,n_estimators)
    frame_E = LabelFrame(win,text='Machine learning results',font = "Arial 14 bold")
    frame_E.grid(row=1,column=0, pady=1,padx=660)
    correct_ExtraTrees=(prediction_ExtraTrees['Actual']==prediction_ExtraTrees['Predict']).sum()
    correct_RandomForest=(prediction_RandomForest['Actual']==prediction_RandomForest['Predict']).sum()
    Label(frame_E,text='Random Forest accuracy (training): '+str(round(r2_train_RandomForest,2)),font = "Arial 14  ").grid(row=0,sticky='W', padx=5, pady=10) 
    Label(frame_E,text='Random Forest accuracy (test): '+str(round(r2_test_RandomForest,2)),font = "Arial 14  ").grid(row=1,sticky='W', padx=5, pady=10) 
    Label(frame_E,text='Extra Trees accuracy (training): '+str(round(r2_train_ExtraTrees,2)),font = "Arial 14  ").grid(row=3,sticky='W', padx=5, pady=10) 
    Label(frame_E,text='Extra Trees accuracy (test): '+str(round(r2_test_ExtraTrees,2)),font = "Arial 14  ").grid(row=4,sticky='W', padx=5, pady=10) 
    Label(frame_E,text='Average of cross validation (RF): '+str(round(np.mean(cvrf),2)),font = "Arial 14  ").grid(row=6,sticky='W', padx=5, pady=10) 
    Label(frame_E,text='Average of cross validation (ET): '+str(round(np.mean(cvxt),2)),font = "Arial 14  ").grid(row=7,sticky='W', padx=5, pady=10) 
    Label(frame_E,text='Correct prediction by ExtraTrees= '+str(correct_ExtraTrees)+' OUT OF '+str(len(X_test)),font = "Arial 14  ").grid(row=5,sticky='W', padx=5, pady=10) 
    Label(frame_E,text='Correct prediction by RandomForest= '+str(correct_RandomForest)+' OUT OF '+str(len(X_test)),font = "Arial 14  ").grid(row=2,sticky='W', padx=5, pady=10)  
    feature_name_ExtraTrees=pd.DataFrame(feature_name_ExtraTrees)
    feature_name_RandomForest=pd.DataFrame(feature_name_RandomForest)
    feature_name_X_selected=pd.DataFrame(feature_name_X_selected)
    ET_corr = Button(master =frame_E, command =lambda:et_corr(),text ='Plot Extra Trees correlation') 
    ET_corr .grid(row=9, column=0, sticky='W', padx=1, pady=3) 
    RF_corr = Button(master =frame_E, command =lambda:rf_corr(),text ='Plot Random Forest correlation') 
    RF_corr .grid(row=8, column=0, sticky='W', padx=1, pady=3) 
    df3 = pd.merge(feature_name_ExtraTrees,feature_name_RandomForest, how='inner')
    
    def corr_plot(typeplt):
        global plotplot
        import seaborn as sns
        pl=Tk()
        screen_width=1440
        screen_height=900 
        pl.geometry(f'{screen_width}x{screen_height}')
        a=np.array(corext)
        b=np.array(corrf)
        if typeplt=='et':
            title = pl.title("ExtraTrees Correlation")
            xx=corext
            mina=a.min()
        if typeplt=='rf':
            title = pl.title("RandomForest Correlation")
            xx=corrf
            mina=b.min()
        global plo
        plo= LabelFrame(pl)
        plo.place(x=0,y=0,width='1420',height='880')
                       
        
        f = plt.figure(figsize=(12,9))
        ax = sns.heatmap(
            xx, 
            vmin=mina, vmax=1,
            cmap=sns.diverging_palette(22, 220, n=100),
            square=True
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=28,
            horizontalalignment='right',fontsize=9
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            fontsize=9
        )
        if typeplt=='et': 
            plt.title('Selected features by ExtraTreesClassifier (Correlation matrix)',fontsize=14)
        if typeplt=='rf':
            plt.title('Selected features by RandomForestClassifier (Correlation matrix)',fontsize=14)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
            
        canvas = FigureCanvasTkAgg(f, master =plo)
        canvas.draw()
    
        canvas.get_tk_widget().pack()
    
        toolbarFrame = Frame(master=plo)
        toolbarFrame.pack()    
        toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
    
        canvas._tkcanvas.pack()   
        
        mainloop()     
    machine_learning_same = Button(master =frame_D, command =lambda:machine_run_same(cvv,n_unique,n_estimatorss,X_train, y_train,X_test,y_test,machine_data,X,y),text ='material prediction with same data') 
    machine_learning_same .grid(row=5, column=0, sticky='W', padx=5, pady=3)     
    machine_learning_best = Button(master =frame_D, command =lambda:best(X_train, y_train,X_test,y_test,machine_data,X,y,n,n_estimatorss,k_bests),text ='material prediction with best features') 
    machine_learning_best .grid(row=6, column=0, sticky='W', padx=5, pady=3)    
    Label(frame_D,text='Feature number',font = "Arial 14  ").grid(row=3,sticky='W', padx=5, pady=10) 
    k_bests = Entry(master=frame_D, width=5)
    k_bests.grid(row=3,column=1,sticky='W', padx=2, pady=2)
    k_bests.insert(END, 20)
    def et_corr():
        typeplt='et'
        corr_plot(typeplt)
    def rf_corr():
        typeplt='rf'
        corr_plot(typeplt)
      
        
    def machine_run_same(cvv,n_unique,n_estimatorss,X_train, y_train,X_test,y_test,machine_data,X,y):
        GUI=True
        n_estimators=int(n_estimatorss.get())
        cv=int(cvv.get())
        m=1
        srf=np.zeros(m)
        sxt=np.zeros(m)
        prf=np.zeros(m)
        pxt=np.zeros(m)
        cvrf=np.zeros(m)
        cvxt=np.zeros(m)
        y=   machine_data['MaterialClass']   

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.naive_bayes import CategoricalNB
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import ExtraTreesClassifier   
        from sklearn.feature_selection import SelectFromModel
        mod=  ExtraTreesClassifier(n_estimators=n_estimators, random_state=1).fit(X_train, y_train)
        model_ExtraTreesClassifier= SelectFromModel(mod, prefit=True)
        X_new_ExtraTreesClassifier = model_ExtraTreesClassifier.transform(X)
        feature_idx = model_ExtraTreesClassifier.get_support()
        feature_name_ExtraTrees = X_train.columns[feature_idx]
        #print(X_new_ExtraTreesClassifier.shape)
        score_ExtraTrees = cross_val_score(ExtraTreesClassifier(n_estimators=n_estimators), X_train, y_train, cv=cv)
        #print('\n######Cross validation of etxra trees for '+str(cv)+' folds#######\n')
        #print(score_ExtraTrees)
        feature_importance_extra = mod.feature_importances_
        #print(feature_importance_extra)
        
        #print('\n######Cross validation of etxra trees average#######')
        #print(np.mean(score_ExtraTrees))
        cvxt=np.mean(score_ExtraTrees)
        

        ###############RANDOM FOREST
        
        mode=  RandomForestClassifier(n_estimators=n_estimators, random_state=1).fit(X_train, y_train)
        model_RandomForestClassifier = SelectFromModel(mode, prefit=True)
        X_new_RandomForestClassifier = model_RandomForestClassifier.transform(X)
        feature_idx = model_RandomForestClassifier.get_support()
        feature_name_RandomForest = X_train.columns[feature_idx]
        #print(X_new_RandomForestClassifier.shape)
        feature_importance_random = mode.feature_importances_
        #print(feature_importance_random)
        
        score_RandomForest = cross_val_score(RandomForestClassifier(n_estimators=n_estimators), X_train, y_train, cv=cv)
        #print('\n######Cross validation of random forest for '+str(cv)+' folds#######\n')
        #print(score_RandomForest)
        #print('\n######Cross validation of random forest average#######')
        #print(np.mean(score_RandomForest))
        cvrf=np.mean(score_RandomForest)
        ##############

        y=(y.astype(float))
        
        ##MODEL ACCURACY
        r2_train_RandomForest=mode.score(X_train,y_train)
        r2_test_RandomForest=mode.score(X_test,y_test)
        #print('\n******************')
        #print('Training_Random_Forest: '+str(r2_train_RandomForest))
        #print('Test_Random_Forest: '+str(r2_test_RandomForest))
        srf=r2_test_RandomForest
        prediction_RandomForest=mode.predict(X_test)
        
        prediction_RandomForest=pd.DataFrame(prediction_RandomForest)
        prediction_RandomForest.insert(0,'Actual', y_test.values)
        for i in range (len(prediction_RandomForest)):
            if prediction_RandomForest[0][i]<0:
                prediction_RandomForest[0][i]=0
        prediction_RandomForest[0]=round(prediction_RandomForest[0])
        r2_train_ExtraTrees=mod.score(X_train,y_train)
        r2_test_ExtraTrees=mod.score(X_test,y_test)
        #print('\n******************')
        #print('Training_Extra_Trees: '+str(r2_train_ExtraTrees))
        #print('Test_Extra_Trees: '+str(r2_test_ExtraTrees))
        sxt=r2_test_ExtraTrees
        prediction_ExtraTrees=mod.predict(X_test)
        
        prediction_ExtraTrees=pd.DataFrame(prediction_ExtraTrees)
        prediction_ExtraTrees.insert(0,'Actual', y_test.values)
        for i in range (len(prediction_ExtraTrees)):
            if prediction_ExtraTrees[0][i]<0:
                prediction_ExtraTrees[0][i]=0
                
        prediction_RandomForest = prediction_RandomForest.rename({0: 'Predict'}, axis='columns')
        prediction_ExtraTrees = prediction_ExtraTrees.rename({0: 'Predict'}, axis='columns')
        
        unknown_sample=[2.02,9.76,3,2540,2.94,9.64,1596,0.25,0,0,27,10,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6.91,0,0,0,0,0,0,0,0,0,0,0,0,22.71,0.86,83.16,6.28,642.01,0.21,0,19.87,0,0,35,0,0,13000,2900,0,0,0,0,0,0,0,0,0]
        unknown_sample=pd.DataFrame(unknown_sample)
        unknown_sample=unknown_sample.T
        print('\n#####PREDICTION OF A SAMPLE WITH MATERIAL CLASS 8#####')
        print('\n******PREDICTION OF RANDOM FOREST')
        print(mode.predict(unknown_sample))
        print('\n******PREDICTION OF EXTRA TREES')
        print(mod.predict(unknown_sample))  
        prf=mode.predict(unknown_sample)
        pxt=mod.predict(unknown_sample)
        
        ## CORRELATION 
        
        cor=machine_data
        cor=cor.drop(columns=['PAK', 'KW_Index','Bromacil_Eluate'])
        corr = cor.corr()
        X_new_ExtraTreesClassifier=pd.DataFrame(X_new_ExtraTreesClassifier)
        X_new_RandomForestClassifier=pd.DataFrame(X_new_RandomForestClassifier)
        for i in range (len(X_new_ExtraTreesClassifier)):
            try:
                X_new_ExtraTreesClassifier = X_new_ExtraTreesClassifier.rename({i: feature_name_ExtraTrees[i]}, axis='columns')
            except:
                pass
        for i in range (len(X_new_RandomForestClassifier)):
            try:
                X_new_RandomForestClassifier = X_new_RandomForestClassifier.rename({i: feature_name_RandomForest[i]}, axis='columns')
            except:
                pass    
        X_new_ExtraTreesClassifier.insert(0,'MaterialClass', y.values)
        X_new_RandomForestClassifier.insert(0,'MaterialClass', y.values)  
        corext = X_new_ExtraTreesClassifier.corr()
        corrf = X_new_RandomForestClassifier.corr()
        from sklearn.feature_selection import SelectKBest

        from sklearn.feature_selection import f_regression
        k=25
        fs = SelectKBest(score_func=f_regression, k=25)
        X_selected = fs.fit_transform(X, y)
        X_selected=pd.DataFrame(X_selected)
        #print(X_selected.shape)
        feature_idx = fs.get_support()
        feature_name_X_selected = X_train.columns[feature_idx]
        for i in range (k):
            try:
                X_selected = X_selected.rename({i: feature_name_X_selected[i]}, axis='columns')
            except:
                pass
        X_selected.insert(0,'MaterialClass', y.values) 
        correlation_nomodel=X_selected.corr()
        frame_E = LabelFrame(win,text='Machine learning results',font = "Arial 14 bold")
        frame_E.grid(row=1,column=0, pady=1,padx=590)
        correct_ExtraTrees=(prediction_ExtraTrees['Actual']==prediction_ExtraTrees['Predict']).sum()
        correct_RandomForest=(prediction_RandomForest['Actual']==prediction_RandomForest['Predict']).sum()
        Label(frame_E,text='Random Forest accuracy (training): '+str(round(r2_train_RandomForest,2)),font = "Arial 14  ").grid(row=0,sticky='W', padx=5, pady=10) 
        Label(frame_E,text='Random Forest accuracy (test): '+str(round(r2_test_RandomForest,2)),font = "Arial 14  ").grid(row=1,sticky='W', padx=5, pady=10) 
        Label(frame_E,text='Extra Trees accuracy (training): '+str(round(r2_train_ExtraTrees,2)),font = "Arial 14  ").grid(row=3,sticky='W', padx=5, pady=10) 
        Label(frame_E,text='Extra Trees accuracy (test): '+str(round(r2_test_ExtraTrees,2)),font = "Arial 14  ").grid(row=4,sticky='W', padx=5, pady=10) 
        Label(frame_E,text='Average of cross validation (RF): '+str(round(np.mean(cvrf),2)),font = "Arial 14  ").grid(row=6,sticky='W', padx=5, pady=10) 
        Label(frame_E,text='Average of cross validation (ET): '+str(round(np.mean(cvxt),2)),font = "Arial 14  ").grid(row=7,sticky='W', padx=5, pady=10) 
        Label(frame_E,text='Correct prediction by ExtraTrees= '+str(correct_ExtraTrees)+' OUT OF '+str(len(X_test)),font = "Arial 14  ").grid(row=5,sticky='W', padx=5, pady=10) 
        Label(frame_E,text='Correct prediction by RandomForest= '+str(correct_RandomForest)+' OUT OF '+str(len(X_test)),font = "Arial 14  ").grid(row=2,sticky='W', padx=5, pady=10)  
        feature_name_ExtraTrees=pd.DataFrame(feature_name_ExtraTrees)
        feature_name_RandomForest=pd.DataFrame(feature_name_RandomForest)
        feature_name_X_selected=pd.DataFrame(feature_name_X_selected)
        ET_corr = Button(master =frame_E, command =lambda:et_corr(),text ='Plot Extra Trees correlation') 
        ET_corr .grid(row=9, column=0, sticky='W', padx=1, pady=3) 
        RF_corr = Button(master =frame_E, command =lambda:rf_corr(),text ='Plot Random Forest correlation') 
        RF_corr .grid(row=8, column=0, sticky='W', padx=1, pady=3) 
        df3 = pd.merge(feature_name_ExtraTrees,feature_name_RandomForest, how='inner')
        
        def corr_plot(typeplt):
            global plotplot
            import seaborn as sns
            pl=Tk()
            screen_width=1440
            screen_height=900 
            pl.geometry(f'{screen_width}x{screen_height}')
            a=np.array(corext)
            b=np.array(corrf)
            if typeplt=='et':
                title = pl.title("ExtraTrees Correlation")
                xx=corext
                mina=a.min()
            if typeplt=='rf':
                title = pl.title("RandomForest Correlation")
                xx=corrf
                mina=b.min()
            global plo
            plo= LabelFrame(pl)
            plo.place(x=0,y=0,width='1420',height='880')
                        
            
            f = plt.figure(figsize=(12, 9))
            ax = sns.heatmap(
                xx, 
                vmin=mina, vmax=1,
                cmap=sns.diverging_palette(22, 220, n=100),
                square=True
            )
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=28,
                horizontalalignment='right',fontsize=9
            )
            ax.set_yticklabels(
                ax.get_yticklabels(),
                fontsize=9
            )
            if typeplt=='et': 
                plt.title('Selected features by ExtraTreesClassifier (Correlation matrix)',fontsize=14)
            if typeplt=='rf':
                plt.title('Selected features by RandomForestClassifier (Correlation matrix)',fontsize=14)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=10)
                
            canvas = FigureCanvasTkAgg(f, master =plo)
            canvas.draw()
        
            canvas.get_tk_widget().pack()
        
            toolbarFrame = Frame(master=plo)
            toolbarFrame.pack()    
            toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
        
            canvas._tkcanvas.pack()   
            mainloop()     
        def et_corr():
            typeplt='et'
            corr_plot(typeplt)
        def rf_corr():
            typeplt='rf'
            corr_plot(typeplt)
    def best(X_train, y_train,X_test,y_test,machine_data,X,y,n,n_estimatorss,k_bests):    
        
        n_estimators=int(n_estimatorss.get())
        k_best=int(k_bests.get())
        cv=int(cvv.get())
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        import pandas as pd
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.naive_bayes import CategoricalNB
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import ExtraTreesClassifier   
        from sklearn.feature_selection import SelectFromModel
        from sklearn.ensemble import GradientBoostingClassifier
        KBest = SelectKBest(score_func = chi2, k = k_best)
        KBest = KBest.fit(X,y)
        df_scores = pd.DataFrame({'features': X.columns, 'Chi2Score': KBest.scores_, 'pValue': KBest.pvalues_ })
        global mc
        mc=machine_data
        colschi = KBest.get_support(indices=True)
        featureschi = X.columns[colschi]
        machine_data=machine_data[featureschi]
        machine_data.insert(0,'MaterialClass', y.values)
        machine_data.insert(0,'Probe', mc['Probe'])
        
        for j in range (len(srf)):
            machine_data=mc
            ##Take unique sample
            unique=machine_data['Probe'].unique()
            unique=pd.DataFrame(unique)
            unique=unique.iloc[1:]
            
            ##Seperate 70 unique samples (NOT 70 DATAPOINTS) for test
            uni_take=unique.sample(n = n)
            XX=pd.DataFrame()
            
            for i in range (len(uni_take)):
                XX=XX.append(machine_data.loc[machine_data['Probe']==uni_take[0].values[i]])
            ##all data from 70 selected samples    
            test_unique=XX    
            ##training samples after seperation of the testing samples
            train_unique = pd.concat([test_unique,machine_data]).drop_duplicates(keep=False)
            test_unique=test_unique.drop(columns=['Probe'])
            train_unique=train_unique.drop(columns=['Probe'])

            machine_data=machine_data.drop(columns=['Probe'])
            X_train=X_train[X.columns[colschi]]
            X_test=X_test[X.columns[colschi]]
            #Since for TotalContentOfSolutes_Eluate value of zeros is meaningless eliminate it from data
            try:
                X_train=X_train.drop(columns=['MaterialClass'])
                X_test=X_test.drop(columns=['MaterialClass'])
            except:
                pass    

            
            #normolize X train and test
            # scaler=MinMaxScaler()
            # X_train_scaled=scaler.fit_transform(X_train)
            # X_test_scaled=scaler.transform(X_test) 
            X=machine_data[X.columns[colschi]]
            #X=X.drop(columns=['MaterialClass'])
            y=   machine_data['MaterialClass']
            
            X_test=(X_test.astype(float))
            X_train=(X_train.astype(float))
            X=(X.astype(float))
            y=(y.astype(float))
            ############EXTRA TREES
            mod=  ExtraTreesClassifier(n_estimators=n_estimators, random_state=1).fit(X_train, y_train).fit(X_train, y_train)
            model_ExtraTreesClassifier= SelectFromModel(mod, prefit=True)
            X_new_ExtraTreesClassifier = model_ExtraTreesClassifier.transform(X)
            feature_idx = model_ExtraTreesClassifier.get_support()
            feature_name_ExtraTrees = X_train.columns[feature_idx]
            score_ExtraTrees = cross_val_score(ExtraTreesClassifier(n_estimators=10), X, y, cv=10)
            print('\n######Cross validation of etxra trees for 10 folds#######\n')
            print(score_ExtraTrees)
            
            print('\n######Cross validation of etxra trees average#######')
            print(np.mean(score_ExtraTrees))
            cvxt[j]=np.mean(score_ExtraTrees)
            

            ###############RANDOM FOREST
            
            mode=  RandomForestClassifier(n_estimators=n_estimators, random_state=1).fit(X_train, y_train).fit(X_train, y_train)
            model_RandomForestClassifier = SelectFromModel(mode, prefit=True)
            X_new_RandomForestClassifier = model_RandomForestClassifier.transform(X)
            feature_idx = model_RandomForestClassifier.get_support()
            feature_name_RandomForest = X_train.columns[feature_idx]
            
            score_RandomForest = cross_val_score(RandomForestClassifier(n_estimators=10), X, y, cv=10)
            print('\n######Cross validation of random forest 10 folds#######\n')
            print(score_RandomForest)
            print('\n######Cross validation of random forest average#######')
            print(np.mean(score_RandomForest))
            cvrf[j]=np.mean(score_RandomForest)
            ##############
            y_train=(y_train.astype(int))
            y_test=(y_test.astype(int))
            
            ##MODEL ACCURACY
            r2_train_RandomForest=mode.score(X_train,y_train)
            r2_test_RandomForest=mode.score(X_test,y_test)
            print('\n******************')
            print('Training_Random_Forest: '+str(r2_train_RandomForest))
            print('Test_Random_Forest: '+str(r2_test_RandomForest))
            srf[j]=r2_test_RandomForest
            prediction_RandomForest=mode.predict(X_test)
            
            prediction_RandomForest=pd.DataFrame(prediction_RandomForest)
            prediction_RandomForest.insert(0,'Actual', y_test.values)
            for i in range (len(prediction_RandomForest)):
                if prediction_RandomForest[0][i]<0:
                    prediction_RandomForest[0][i]=0
            prediction_RandomForest[0]=round(prediction_RandomForest[0])
            r2_train_ExtraTrees=mod.score(X_train,y_train)
            r2_test_ExtraTrees=mod.score(X_test,y_test)
            print('\n******************')
            print('Training_Extra_Trees: '+str(r2_train_ExtraTrees))
            print('Test_Extra_Trees: '+str(r2_test_ExtraTrees))
            sxt[j]=r2_test_ExtraTrees
            prediction_ExtraTrees=mod.predict(X_test)
            
            prediction_ExtraTrees=pd.DataFrame(prediction_ExtraTrees)
            prediction_ExtraTrees.insert(0,'Actual', y_test.values)
            for i in range (len(prediction_ExtraTrees)):
                if prediction_ExtraTrees[0][i]<0:
                    prediction_ExtraTrees[0][i]=0
                    
            prediction_RandomForest = prediction_RandomForest.rename({0: 'Predict'}, axis='columns')
            prediction_ExtraTrees = prediction_ExtraTrees.rename({0: 'Predict'}, axis='columns')

            unknown_sample=[2.02,9.76,3,2540,2.94,9.64,1596,0.25,0,0,27,10,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6.91,0,0,0,0,0,0,0,0,0,0,0,0,22.71,0.86,83.16,6.28,642.01,0.21,0,19.87,0,0,35,0,0,13000,2900,0,0,0,0,0,0,0,0,0]
            unknown_sample=pd.DataFrame(unknown_sample)
            unknown_sample=unknown_sample.T
            unknown_sample=unknown_sample[colschi]
            print('\n#####PREDICTION OF A SAMPLE WITH MATERIAL CLASS 8#####')
            print('\n******PREDICTION OF RANDOM FOREST')
            print(mode.predict(unknown_sample))
            print('\n******PREDICTION OF EXTRA TREES')
            print(mod.predict(unknown_sample))  
            X_new_ExtraTreesClassifier=pd.DataFrame(X_new_ExtraTreesClassifier)
            X_new_RandomForestClassifier=pd.DataFrame(X_new_RandomForestClassifier)
            for i in range (len(X_new_ExtraTreesClassifier)):
                try:
                    X_new_ExtraTreesClassifier = X_new_ExtraTreesClassifier.rename({i: feature_name_ExtraTrees[i]}, axis='columns')
                except:
                    pass
            for i in range (len(X_new_RandomForestClassifier)):
                try:
                    X_new_RandomForestClassifier = X_new_RandomForestClassifier.rename({i: feature_name_RandomForest[i]}, axis='columns')
                except:
                    pass    
            X_new_ExtraTreesClassifier.insert(0,'MaterialClass', y.values)
            X_new_RandomForestClassifier.insert(0,'MaterialClass', y.values)  
            corext = X_new_ExtraTreesClassifier.corr()
            corrf = X_new_RandomForestClassifier.corr()
            frame_E = LabelFrame(win,text='Machine learning results',font = "Arial 14 bold")
            frame_E.grid(row=1,column=0, pady=1,padx=590)
            correct_ExtraTrees=(prediction_ExtraTrees['Actual']==prediction_ExtraTrees['Predict']).sum()
            correct_RandomForest=(prediction_RandomForest['Actual']==prediction_RandomForest['Predict']).sum()
            Label(frame_E,text='Random Forest accuracy (training): '+str(round(r2_train_RandomForest,2)),font = "Arial 14  ").grid(row=0,sticky='W', padx=5, pady=10) 
            Label(frame_E,text='Random Forest accuracy (test): '+str(round(r2_test_RandomForest,2)),font = "Arial 14  ").grid(row=1,sticky='W', padx=5, pady=10) 
            Label(frame_E,text='Extra Trees accuracy (training): '+str(round(r2_train_ExtraTrees,2)),font = "Arial 14  ").grid(row=3,sticky='W', padx=5, pady=10) 
            Label(frame_E,text='Extra Trees accuracy (test): '+str(round(r2_test_ExtraTrees,2)),font = "Arial 14  ").grid(row=4,sticky='W', padx=5, pady=10) 
            Label(frame_E,text='Average of cross validation (RF): '+str(round(np.mean(cvrf),2)),font = "Arial 14  ").grid(row=6,sticky='W', padx=5, pady=10) 
            Label(frame_E,text='Average of cross validation (ET): '+str(round(np.mean(cvxt),2)),font = "Arial 14  ").grid(row=7,sticky='W', padx=5, pady=10) 
            Label(frame_E,text='Correct prediction by ExtraTrees= '+str(correct_ExtraTrees)+' OUT OF '+str(len(X_test)),font = "Arial 14  ").grid(row=5,sticky='W', padx=5, pady=10) 
            Label(frame_E,text='Correct prediction by RandomForest= '+str(correct_RandomForest)+' OUT OF '+str(len(X_test)),font = "Arial 14  ").grid(row=2,sticky='W', padx=5, pady=10)  
            ET_corr = Button(master =frame_E, command =lambda:et_corr(),text ='Plot Extra Trees correlation') 
            ET_corr .grid(row=9, column=0, sticky='W', padx=1, pady=3) 
            RF_corr = Button(master =frame_E, command =lambda:rf_corr(),text ='Plot Random Forest correlation') 
            RF_corr .grid(row=8, column=0, sticky='W', padx=1, pady=3) 
            def corr_plot(typeplt):
                global plotplot
                import seaborn as sns
                pl=Tk()
                screen_width=1440
                screen_height=900 
                pl.geometry(f'{screen_width}x{screen_height}')
                a=np.array(corext)
                b=np.array(corrf)
                if typeplt=='et':
                    title = pl.title("ExtraTrees Correlation")
                    xx=corext
                    mina=a.min()
                if typeplt=='rf':
                    title = pl.title("RandomForest Correlation")
                    xx=corrf
                    mina=b.min()
                global plo
                plo= LabelFrame(pl)
                plo.place(x=0,y=0,width='1420',height='880')
                            
                
                f = plt.figure(figsize=(12, 9))
                ax = sns.heatmap(
                    xx, 
                    vmin=mina, vmax=1,
                    cmap=sns.diverging_palette(22, 220, n=100),
                    square=True
                )
                ax.set_xticklabels(
                    ax.get_xticklabels(),
                    rotation=28,
                    horizontalalignment='right',fontsize=9
                )
                ax.set_yticklabels(
                    ax.get_yticklabels(),
                    fontsize=9
                )
                if typeplt=='et': 
                    plt.title('Selected features by ExtraTreesClassifier (Correlation matrix)',fontsize=14)
                if typeplt=='rf':
                    plt.title('Selected features by RandomForestClassifier (Correlation matrix)',fontsize=14)
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=10)
                    
                canvas = FigureCanvasTkAgg(f, master =plo)
                canvas.draw()
            
                canvas.get_tk_widget().pack()
            
                toolbarFrame = Frame(master=plo)
                toolbarFrame.pack()    
                toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
            
                canvas._tkcanvas.pack()   
                mainloop()     
            def et_corr():
                typeplt='et'
                corr_plot(typeplt)
            def rf_corr():
                typeplt='rf'
                corr_plot(typeplt)

    
        

mainloop()

