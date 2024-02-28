import tkinter
import customtkinter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from DEDlib import *
import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)

from tempfile import TemporaryFile
import pickle

import json
from json import JSONEncoder

matplotlib.rc('axes',edgecolor='white')

def counter():#N,Nmax
    global paused,stopped,started,DEDargs,pbar,Nfin
    #print(count_var.get())
    if not paused and pbar.n!=DEDargs[0]:
        #N+=1
        #pbar.n=N
        #pbar.refresh()
        iterationDED()
        count_var.set(f"{pbar.n}/{DEDargs[0]}")
        progressbar.set(pbar.n/DEDargs[0])
    if not stopped and pbar.n!=DEDargs[0]:
        #pbar.refresh()
        app.after(1,counter)
    elif pbar.n==DEDargs[0]:
        pbar.close()

    else:
        #stopped=False
        started=False
        count_var.set(f"{0}/{DEDargs[0]}")
        progressbar.set(0)
        pbar.n=0
        #Nfin=np.zeros(len(DEDargs[11]),dtype='float')
        #pbar.refresh()
        pbar.reset()

def startDED():
    global started,paused,number,istar,pbar,stopped
    istar+=1
    if istar==1:
        if not started: 
            started=True
            stopped=False
            #pbar.reset()
            pbar.start_t = pbar._time()
            #print("start")
        counter()


def pauseDED():
    global paused,started
    if started:
        paused = not paused 

def stopDED():
    global stopped,started,paused,number,istar,omega,dirsav
    if started or stopped:
        try:
            if not dirsav.get().endswith(".json"):
                dirsav.delete(0,last_index=1000)
                dirsav.insert(0, 'Try again')
            else:   
                AvgSigmajsonfileW(dirsav.get())
        except IOError:
            savfilename()
    if not stopped and started:
        stopped=True
        paused=False
        #print("stop")
        number=0
        istar=0

def fileloader():
    global started,dir,started,omega,AvgSigmadat,Nfin,count_var,progressbar,Lor,DEDargs,eta,pbar,Npoles,Hn,n,nd,c
    if not started:
        try:
            data=AvgSigmajsonfileR(dir.get())
            Nfin,omega,AvgSigmadat,nd=np.array(data["Nfin"]),np.array(data["omega"]),np.array(data["AvgSigmadat"]*data["Nfin"]).squeeze(),np.array(data["nd"])
            DEDargs[1:]=[data["poles"],data["U"],data["Sigma"],data["Ed"],data["Gamma"],data["ctype"],data["Edcalc"],data["Nimpurities"],data["U2"],data["J"],data["Tk"],data["etaco"]]
            #print(Nfin,omega,AvgSigmadat)
            count_var.set(f"{pbar.n}/{DEDargs[0]}")
            progressbar.set(pbar.n/DEDargs[0])
            Lor=Lorentzian(omega,DEDargs[5],DEDargs[1],DEDargs[4],DEDargs[3])[0]
            Npoles=int(DEDargs[1]/DEDargs[8])
            c,eta=[Jordan_wigner_transform(i,2*DEDargs[1]) for i in range(2*DEDargs[1])],DEDargs[12][0]*abs(omega)+DEDargs[12][1]
            (Hn,n)=Operators(c,DEDargs[8],DEDargs[1])
            pbar.n=int(min(Nfin))
            pbar.refresh()
        except IOError:
            dir.delete(0,last_index=1000)
            dir.insert(0, 'Try again')
            #dir.set('Try again')

def savfilename():
    global dirsav
    if not dirsav.get().endswith(".json"):
        dirsav.delete(0,last_index=1000)
        dirsav.insert(0, 'Try again')
        #dirsav.set('Try again')


def AvgSigmajsonfileR(name):#dir
    #text_file=open(dir,"r")
    #lines=text_file.read().split('\n')
    #text_file.close()
    #return np.array([np.array(l,dtype=object).astype(np.complex) for l in [lines[i].split('\t') for i, _ in enumerate(lines[1:])]]).T
    data=json.load(open(name))
    data["AvgSigmadat"]=np.array(data["AvgSigmadat"],dtype=object).astype(np.complex128)
    return data

def AvgSigmajsonfileW(name):
    global omega,AvgSigmadat,Nfin,DEDargs,nd
    #np.savetxt(name+'.txt',np.c_[omega,np.real((AvgSigmadat/Nfin[:,None]).squeeze()),np.imag((AvgSigmadat/Nfin[:,None]).squeeze())],fmt='%.18f\t(%.18g%+.18gj)',delimiter='\t',newline='\n')

    #outfile = TemporaryFile()
    #np.savez(outfile, Nfin=Nfin,omega=omega,AvgSigma=(AvgSigmadat/Nfin[:,None]).squeeze())

    #data={"Nfin": Nfin,"omega": omega,"AvgSigmadatreal":np.real((AvgSigmadat/Nfin[:,None]).squeeze()),"AvgSigmadatimag":np.imag((AvgSigmadat/Nfin[:,None]).squeeze())}
    

    #poles,U,Sigma,Ed,Gamma,ctype,Edcalc,Nimpurities,U2,J,Tk,etaco
    
    data={"poles": DEDargs[1],"U": DEDargs[2],"Sigma": DEDargs[3],"Ed": DEDargs[4],"Gamma": DEDargs[5],"ctype": DEDargs[6],"Edcalc": DEDargs[7],
    "Nimpurities": DEDargs[8],"U2": DEDargs[9],"J": DEDargs[10],"Tk": DEDargs[11],"etaco": DEDargs[12],
    "Nfin": Nfin,"omega": omega,"AvgSigmadat":[str(i) for i in (AvgSigmadat/Nfin[:,None]).squeeze()],"nd": np.real(nd/Nfin).squeeze()}
    jsonObj=json.dumps(data, cls=NumpyArrayEncoder)
    with open(name, "w") as outfile:
        outfile.write(jsonObj)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def valinit():
    global omega,selectpcT,selectpT,Npoles,c,pbar,eta,Hn,n,AvgSigmadat,Nfin,nd,DEDargs,Lor
    if log: omega,selectpcT,selectpT,Npoles=np.concatenate((-np.logspace(np.log(bound)/np.log(base),np.log(1e-5)/np.log(base),int(np.round(SizeO/2)),base=base),np.logspace(np.log(1e-5)/np.log(base),np.log(bound)/np.log(base),int(np.round(SizeO/2)),base=base))),[],[],int(poles/Nimpurities)
    else: omega,selectpcT,selectpT,Npoles=np.linspace(-bound,bound,SizeO),[],[],int(poles/Nimpurities)
    c,pbar,eta=[Jordan_wigner_transform(i,2*poles) for i in range(2*poles)],trange(N,position=posb,leave=False,desc='Iterations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),etaco[0]*abs(omega)+etaco[1]
    (Hn,n),AvgSigmadat,Nfin,nd=Operators(c,Nimpurities,poles),np.zeros((len(Tk),SizeO),dtype='complex_'),np.zeros(len(Tk),dtype='float'),np.zeros(len(Tk),dtype='complex_')
    Lor=Lorentzian(omega,DEDargs[5],DEDargs[1],DEDargs[4],DEDargs[3])[0]
    #add stop inputs from changing if file is already loaded

def DOSplottest(fDOS,Lor,omega,name,labels,log=False,ymax=1.2,save=True):
    """DOSplot(fDOS,Lor,omega,name,labels). 
A plot function to present results from the AIM moddeling for a single results with a comparison to the non-interacting DOS."""
    fig=plt.figure(figsize=(10,8))
    plt.rc('legend',fontsize=17)
    plt.rc('font',size=25)
    plt.rc('xtick',labelsize=25)
    plt.rc('ytick',labelsize=25)
    axis_font={'fontname':'Calibri','size':'25'}
    plt.xlim(min(omega),max(omega))
    if not log:
        plt.gca().set_ylim(bottom=0,top=ymax)
        plt.gca().set_xticks(np.linspace(min(omega),max(omega),2*int(max(omega))+1),minor=False)
    else: 
        plt.yscale('log')
        plt.gca().set_ylim(bottom=0.0001,top=10)
        plt.gca().set_xticks([-8,-6,-4,-2,0,2,4,6,8],minor=False)
    plt.xlabel("$\\omega$ [-]",**axis_font)
    plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom",rotation=0,labelpad=30,**axis_font)
    plt.plot(omega,Lor,'--r',linewidth=4,label='$\\rho_0$')
    plt.plot(omega,fDOS,'-b',label=labels)
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(name+'.png',format='png')
        plt.savefig(name+'.svg',format='svg',dpi=3600)
    plt.pause(5)
    plt.close()
    return fig

def showgraph():
    global omega,Lor,AvgSigmadat,Nfin,DEDargs,app
    fDOS=(-np.imag(np.nan_to_num(1/(omega-AvgSigmadat/Nfin[:,None]-DEDargs[4]+1j*DEDargs[5])))/np.pi).squeeze()
    #fig=DOSplottest(fDOS,Lor,omega,'constraintN5p','$\\rho,$n='+str(DEDargs[1]),save=False)
    fig=plt.figure(figsize=(7,5.6))#,facecolor='#242424')

    plt.rc('legend',fontsize=12)
    plt.rc('font',size=17)
    plt.rc('xtick',labelsize=17,color='white')
    plt.rc('ytick',labelsize=17,color='white')
    axis_font={'fontname':'Calibri','size':'17'}
    plt.xlim(min(omega),max(omega))
    plt.gca().set_ylim(bottom=0,top=1.2)
    plt.gca().set_xticks(np.linspace(min(omega),max(omega),2*int(max(omega))+1),minor=False)
    plt.xlabel("$\\omega$ [-]",**axis_font,color='white')
    plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom",rotation=0,labelpad=30,**axis_font,color='white')
    plt.plot(omega,Lor,'--r',linewidth=4,label='$\\rho_0$')
    plt.plot(omega,fDOS,'-b',label='$\\rho,$n='+str(DEDargs[1]))
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    fig.set_facecolor("none")

    #ax = plt.axes()

    plt.gca().set_facecolor("#242424")

    canvas = FigureCanvasTkAgg(fig,master=app)

    s = tkinter.ttk.Style()
    bg = s.lookup("TFrame", "background")
    bg_16bit =app.winfo_rgb('DarkGray')#bg
    bg_string="#242424"

    #bg_string = "#" + "".join([hex(bg_color >> 8)[2:] for bg_color in bg_16bit])
    canvas.get_tk_widget().config(bg=bg_string)

    canvas.draw()
    canvas.get_tk_widget().grid(row=8,columnspan=3)#.get_tk_widget()
    app.update()

#global paused,started,stopped
#paused=False
#started=False
#stopped=False
#istar=0
#number=0

# System Settings
#customtkinter.set_appearance_mode("dark")#"System"
#customtkinter.set_default_color_theme("blue")

# Our app frame
#app=customtkinter.CTk()
#app.geometry("500x300")#("720x480")
#app.minsize(300, 200)
#app.title("Distributional Exact Diagonalization AIM simulator")

# Adduing UI Elements
#title=customtkinter.CTkLabel(app,text="Choose DED data file directory")
#title.pack(padx=10,pady=10)
#title.grid(row=0, column=0,columnspan=3)

# Directory input
#dir_var=tkinter.StringVar()
#dir=customtkinter.CTkEntry(app, width=350,height=40,textvariable=dir_var)
#dir.pack()
#dir.grid(row=1, column=0,columnspan=3)


#progressbar=customtkinter.CTkProgressBar(app)
#progressbar.pack(padx=10,pady=10,side = "left")
#progressbar.grid(row=2, column=0,padx=10,pady=10, columnspan=2, sticky="nsew")
#progressbar.set(0)

# Count display
#count_var=tkinter.StringVar()
#count_var.set(f"{number}/{Nmax}")#
#count_text=customtkinter.CTkLabel(app,textvariable=count_var)
#count_text.pack(side = "left")
#count_text.grid(row=2, column=2,padx=10,pady=10, sticky="nsew")

# Start and pause button
#startb=customtkinter.CTkButton(app,text="Start",command=startDED)#command=lambda: startDED(started))
#startb.pack(padx=10,pady=10,side = "left")
#startb.grid(row=3, column=0,padx=10,pady=10, sticky="ew")

#pauseb=customtkinter.CTkButton(app,text="Pause",command=pauseDED)#command=lambda: pauseDED(paused,started))
#pauseb.pack(padx=10,pady=10,side="left")
#pauseb.grid(row=3, column=1,padx=10,pady=10, sticky="ew")

#stopb=customtkinter.CTkButton(app,text="Stop",command=stopDED)#command=lambda: stopDED(started))
#stopb.pack(padx=10,pady=10,side = "left")
#stopb.grid(row=3, column=2,padx=10,pady=10, sticky="ew")

#print(startb["state"])


# Run app

def DEDUI():
    global paused,started,stopped,istar,number,count_var,progressbar,dir,app,DEDargs,dirsav
    paused=False
    started=False
    stopped=False
    istar=0
    number=0
    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("blue")
    app=customtkinter.CTk()
    app.geometry("600x800")
    app.minsize(400, 200)

    #app.resizable(0, 0)

    app.columnconfigure(0, weight=1)
    #app.rowconfigure(0, weight=1)

    app.after(201, lambda :app.iconbitmap('DEDicon.ico'))

    app.title("Distributional Exact Diagonalization AIM simulator")
    title=customtkinter.CTkLabel(app,anchor="w",text="Choose DED data file directory to load")#,anchor="w"
    title.grid(row=0, column=0,columnspan=14,padx=10)
    dir=customtkinter.CTkEntry(app,width=500,height=40,placeholder_text="C:\\")#,textvariable=dir_var
    dir.grid(row=1, column=0,columnspan=14,padx=10)
    loadb=customtkinter.CTkButton(app,text="Submit",width=60,command=fileloader)
    loadb.grid(row=1, column=14,padx=10,pady=10)
    title2=customtkinter.CTkLabel(app,anchor="w",text="Choose DED data file save name")
    title2.grid(row=2, column=0,columnspan=14,padx=10)
    dirsav=customtkinter.CTkEntry(app, width=500,height=40,placeholder_text="example.json")
    dirsav.grid(row=3, column=0,columnspan=14,padx=10)
    saveb=customtkinter.CTkButton(app,text="Submit",width=60,command=savfilename)
    saveb.grid(row=3, column=14,padx=10,pady=10)

    Ntitle=customtkinter.CTkLabel(app,text="N",width=5)
    Ntitle.grid(row=4, column=0)
    Nval=customtkinter.CTkEntry(app, width=80,height=30,placeholder_text="200000")
    Nval.grid(row=4, column=1,padx=5)
    polestitle=customtkinter.CTkLabel(app,text="poles")
    polestitle.grid(row=4, column=3)
    polesval=customtkinter.CTkEntry(app, width=30,height=30,placeholder_text="2")
    polesval.grid(row=4, column=4,padx=5)
    Utitle=customtkinter.CTkLabel(app,text="U")
    Utitle.grid(row=4, column=5)
    Uval=customtkinter.CTkEntry(app, width=30,height=30,placeholder_text="3")
    Uval.grid(row=4, column=6,padx=5)
    Sigmatitle=customtkinter.CTkLabel(app,text="Sigma")
    Sigmatitle.grid(row=4, column=7)
    Sigmaval=customtkinter.CTkEntry(app, width=40,height=30,placeholder_text="1.5")
    Sigmaval.grid(row=4,column=8,padx=5)
    Edtitle=customtkinter.CTkLabel(app,text="Ed")
    Edtitle.grid(row=4, column=9)
    Edval=customtkinter.CTkEntry(app, width=40,height=30,placeholder_text="-1.5")
    Edval.grid(row=4,column=10,padx=5)
    Gammatitle=customtkinter.CTkLabel(app,text="Gamma")
    Gammatitle.grid(row=4, column=11)
    Gammaval=customtkinter.CTkEntry(app, width=50,height=30,placeholder_text="0.3")
    Gammaval.grid(row=4,column=12,padx=5)
    inputb=customtkinter.CTkButton(app,text="Submit",width=60,command=valinit)
    inputb.grid(row=3, column=14,padx=10,pady=10)
    progressbar=customtkinter.CTkProgressBar(app)
    progressbar.grid(row=5, column=0,padx=10,pady=10, columnspan=2, sticky="nsew")
    progressbar.set(0)
    count_var=tkinter.StringVar()
    count_var.set(f"{number}/{DEDargs[0]}")
    count_text=customtkinter.CTkLabel(app,textvariable=count_var)
    count_text.grid(row=5, column=2,padx=10,pady=10, sticky="nsew")
    return app

    """

    progressbar=customtkinter.CTkProgressBar(app)
    progressbar.grid(row=5, column=0,padx=10,pady=10, columnspan=2, sticky="nsew")
    progressbar.set(0)
    count_var=tkinter.StringVar()
    count_var.set(f"{number}/{DEDargs[0]}")
    count_text=customtkinter.CTkLabel(app,textvariable=count_var)
    count_text.grid(row=5, column=2,padx=10,pady=10, sticky="nsew")
    startb=customtkinter.CTkButton(app,text="Start",command=startDED)
    startb.grid(row=6, column=0,padx=10,pady=10, sticky="ew")
    pauseb=customtkinter.CTkButton(app,text="Pause",command=pauseDED)
    pauseb.grid(row=6, column=1,padx=10,pady=10, sticky="ew")
    stopb=customtkinter.CTkButton(app,text="Stop",command=stopDED)
    stopb.grid(row=6, column=2,padx=10,pady=10, sticky="ew")
    stopb=customtkinter.CTkButton(app,text="Show Results",command=showgraph)
    stopb.grid(row=7, column=1,padx=10,pady=10, sticky="ew")

    frame=customtkinter.CTkFrame(app,width=2.2*app.winfo_width(),height=1.8*app.winfo_height())
    frame.grid(row=8,columnspan=3)"""

    #dir_var=tkinter.StringVar()
    #     
    #fig=plt.figure(figsize=(7,5.6))
    #canvas = FigureCanvasTkAgg(fig,master=app)
    #canvas.draw()
    #canvas.get_tk_widget().grid(row=6,columnspan=3)

#nd,_,fDOS,Lor,omega,selectpT,selectpcT,tsim=main(**{"N":200000,"poles":2,"Ed":-3/2,"ctype":'n'})#28 it/s in 3:00

def main(N=200000,poles=2,U=3,Sigma=3/2,Ed=-3/2,Gamma=0.3,SizeO=1001,etaco=[0.02,1e-39],ctype='n',Edcalc='',bound=3,Tk=[0],Nimpurities=1,U2=0,J=0,posb=1,log=False,base=1.5):
    global DEDargs
    #omega,selectpcT,selectpT,Npoles,c,pbar,eta,Hn,n,AvgSigmadat,Nfin,nd,Lor
    DEDargs=[N,poles,U,Sigma,Ed,Gamma,ctype,Edcalc,Nimpurities,U2,J,Tk,etaco,SizeO,bound,posb,log,base]

    DEDUI().mainloop()


def iterationDED(reset=False):
    global omega,selectpcT,selectpT,Npoles,c,pbar,eta,Hn,n,AvgSigmadat,Nfin,nd
    while not reset:
        NewM,nonG,select=Startrans(Npoles,np.sort(Lorentzian(omega,DEDargs[5],Npoles,DEDargs[4],DEDargs[3])[1]),omega,eta)
        H0,H=HamiltonianAIM(np.repeat(NewM[0][0],DEDargs[8]),np.tile([NewM[k+1][k+1] for k in range(len(NewM)-1)],(DEDargs[8],1)),np.tile(NewM[0,1:],(DEDargs[8],1)),DEDargs[2],DEDargs[3],DEDargs[9],DEDargs[10],Hn)
        try: (MBGdat,Boltzmann,Ev0),reset=Constraint(DEDargs[6],H0,H,omega,eta,c,n,DEDargs[11],np.array([ar<DEDargs[0] for ar in Nfin]))
        except (np.linalg.LinAlgError,ValueError,scipy.sparse.linalg.ArpackNoConvergence): (MBGdat,Boltzmann,Ev0),reset=(np.zeros(len(omega),dtype='complex_'),np.zeros(len(DEDargs[11])),np.array([])),False
        if np.isnan(1/nonG-1/MBGdat+DEDargs[3]).any() or np.array([i>=1000 for i in np.real(1/nonG-1/MBGdat+DEDargs[3])]).any(): reset=False
        selectpT.append(select)
    Nfin,AvgSigmadat,nd=Nfin+Boltzmann,AvgSigmadat+(1/nonG-1/MBGdat+DEDargs[3])*Boltzmann[:,None],nd+np.conj(Ev0).T@sum(Hn[0]).data.tocoo()@Ev0*Boltzmann
    selectpcT.append(select)
    if DEDargs[6]=='sn': pbar.n+=1
    else: pbar.n=int(min(Nfin))
    pbar.refresh()
    #print(pbar.n,Boltzmann,Boltzmann+Nfin)
main(N=2000000)#27.5 it/s in 3:00