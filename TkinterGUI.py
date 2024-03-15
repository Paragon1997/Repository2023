## module TkinterGUI
''' TkinterGUI is the Graphical User Interface made with Tkinter for using the DEDlib tooling library'''

import tkinter as tk
import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
from DEDlib import *
import warnings
import json
from ast import literal_eval
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
warnings.filterwarnings("ignore",category=RuntimeWarning)

#rethink poles DOS plotting, not saving all selected poles instead count no. of poles per energy ineterval and save
#add class instance where you can loop multiple simulations without waiting for start GUI but still shows progress and autosave
#status bar in window
#other main functions in seperate windows (graphene, S etc.)
#make exe with pyinstaller

class NumpyArrayEncoder(json.JSONEncoder):
    """``NumpyArrayEncoder(json.JSONEncoder)``.\n
Encodes Numpy array for ``json.dumps()``."""
    def default(self,obj):
        if isinstance(obj,np.ndarray): return obj.tolist()
        return json.JSONEncoder.default(self,obj)

def AvgSigmajsonfileW(root,name):
    """``AvgSigmajsonfileW(root,name)``.\n
Writes ``.json`` file including DED simulation settings and collected data."""
    if root.DEDargs[7]=='AS':root.fDOS=(-np.imag(np.nan_to_num(1/(root.omega-root.AvgSigmadat/root.Nfin[:,None]+(root.AvgSigmadat[:,int(np.round(root.DEDargs[13]/2))]/root.Nfin)[:,None]+1j*root.DEDargs[5])))/np.pi).squeeze()
    else:root.fDOS=(-np.imag(np.nan_to_num(1/(root.omega-root.AvgSigmadat/root.Nfin[:,None]-root.DEDargs[4]+1j*root.DEDargs[5])))/np.pi).squeeze()
    data={"simtype":root.simtype,"Ntot":root.pbar.total,"Nit":root.pbar.n,"telapsed":root.pbar.format_dict["elapsed"],"poles":root.DEDargs[1],"U":root.DEDargs[2],"Sigma":root.DEDargs[3],"Ed":root.DEDargs[4],"Gamma":root.DEDargs[5],"ctype":root.DEDargs[6],"Edcalc":root.DEDargs[7],"Nimpurities":root.DEDargs[8],"U2":root.DEDargs[9],"J":root.DEDargs[10],"Tk":root.DEDargs[11],"etaco":root.DEDargs[12],"SizeO":root.DEDargs[13],"bound":root.DEDargs[14],"posb":root.DEDargs[15],"log":root.DEDargs[16],"base":root.DEDargs[17],
    "Nfin":root.Nfin,"omega":root.omega,"fDOS":root.fDOS,"AvgSigmadat":[str(i) for i in (root.AvgSigmadat/root.Nfin[:,None]).squeeze()],"nd":[str(i) for i in (root.nd/root.Nfin)]}
    with open(name,"w") as outfile: outfile.write(json.dumps(data,cls=NumpyArrayEncoder))

def polesjsonfileW(root,name):
    try:
        root.fDOS=polesfDOS(root.DOSp,root.DEDargs[14],float(root.graphcorrfactor_Entry.get()))
        data={"simtype":root.simtype,"Ntot":root.pbar.total,"Nit":root.pbar.n,"telapsed":root.pbar.format_dict["elapsed"],"poles":root.DEDargs[1],"U":root.DEDargs[2],"Sigma":root.DEDargs[3],"Ed":root.DEDargs[4],"Gamma":root.DEDargs[5],"ctype":root.DEDargs[6],"Edcalc":root.DEDargs[7],"Nimpurities":root.DEDargs[8],"U2":root.DEDargs[9],"J":root.DEDargs[10],"Tk":root.DEDargs[11],"etaco":root.DEDargs[12],"SizeO":root.DEDargs[13],"bound":root.DEDargs[14],"posb":root.DEDargs[15],"log":root.DEDargs[16],"base":root.DEDargs[17],
        "Nfin":root.Nfin,"ratio":int(root.polesratio_Entry.get()),"corrfactor":float(root.graphcorrfactor_Entry.get()),"omega":root.omega,"omegap":root.omegap,"fDOS":root.fDOS,"DOSp":root.DOSp}
        with open(name,"w") as outfile: outfile.write(json.dumps(data,cls=NumpyArrayEncoder))
    except ValueError:pass

def savfilename(root,entry):
    """``savfilename(root,entry)``.\n
Checks whether filename input in entry is valid for use in constructing ``.json`` file."""
    if not entry.get().endswith(".json"):
        entry.delete(0,last_index=tk.END)
        entry.insert(0,'Try again')
    _=savedata(root,entry)

def savedata(root,entry):
    """``savedata(root,entry)``.\n
Saves DED results based on current simulation data."""
    if root.started or root.stopped:
        try:
            if not entry.get().endswith(".json"):
                entry.delete(0,last_index=tk.END)
                entry.insert(0,'Try again')
                return False
            else:   
                {"SAIM":AvgSigmajsonfileW,"poles":polesjsonfileW}[root.simtype](root,entry.get())
                return True
        except IOError:
            entry.delete(0,last_index=tk.END)
            entry.insert(0,'Try again')
            return False
    else: return False

def savegraph(root,entry):
    """``savegraph(root,entry)``.\n
Draws and saves DOS graph including the interacting DOS data acquired from the current DED simulation results."""
    mpl.rc('axes',edgecolor='black')
    _=savedata(root,entry)
    {"SAIM":SAIMgraph,"poles":polesgraph}[root.simtype](root,entry)

def SAIMgraph(root,entry):    
    if root.DEDargs[7]=='AS':root.fDOS=(-np.imag(np.nan_to_num(1/(root.omega-root.AvgSigmadat/root.Nfin[:,None]+(root.AvgSigmadat[:,int(np.round(root.DEDargs[13]/2))]/root.Nfin)[:,None]+1j*root.DEDargs[5])))/np.pi).squeeze()
    else:root.fDOS=(-np.imag(np.nan_to_num(1/(root.omega-root.AvgSigmadat/root.Nfin[:,None]-root.DEDargs[4]+1j*root.DEDargs[5])))/np.pi).squeeze()
    if entry.get().endswith(".json"):
        try:
            DOSplot(root.fDOS,root.Lor,root.omega,entry.get().replace(".json",""),root.graphlegend_Entry.get(),log=bool(root.graphlogy_checkbox.get()),ymax=float(root.graphymax_Entry.get()))
            if root.DEDargs[16]:
                DOSxlogplot(root.fDOS,root.Lor,root.omega,entry.get().replace(".json","")+"logx",root.graphlegend_Entry.get(),ymax=float(root.graphymax_Entry.get()),incneg=True)
                DOSxlogplot(root.fDOS,root.Lor,root.omega,entry.get().replace(".json","")+"logxpos",root.graphlegend_Entry.get(),ymax=float(root.graphymax_Entry.get()),incneg=False)
        except:pass
    else:
        entry.delete(0,last_index=tk.END)
        entry.insert(0,'Try again')

def polesgraph(root,entry):
    try:
        root.fDOS=polesfDOS(root.DOSp,root.DEDargs[14],float(root.graphcorrfactor_Entry.get()))
        root.Lorp=Lorentzian(root.omegap,root.DEDargs[5],root.DEDargs[1],root.DEDargs[4],root.DEDargs[3])[0]
        if entry.get().endswith(".json"):DOSplot(root.fDOS,root.Lorp,root.omegap,entry.get().replace(".json",""),root.graphlegend_Entry.get(),log=bool(root.graphlogy_checkbox.get()),ymax=float(root.graphymax_Entry.get()),bound=root.DEDargs[14])
        else:
            entry.delete(0,last_index=tk.END)
            entry.insert(0,'Try again')
    except:pass

def enableroot(root):
    """``enableroot(root)``.\n
Function to show messagebox when attempting to close ``ctk.CTkToplevel`` window."""
    root.msg=CTkMessagebox(master=root,title="Exit?",message="Do you want to close the program?\nUnsaved progress will be lost.",icon="question",option_1="Cancel",option_2="No",option_3="Yes")
    if root.msg.get()=="Yes":
        root.root.scaling_optionemenu.configure(state="normal")
        root.root.button_open.configure(state="normal")
        root.lift(root.root)
        root.destroy()

def saveasfile(root):
    """``saveasfile(root)``.\n
Function to choose save file location for a DED data file in File Explorer."""
    root.savefile=ctk.filedialog.asksaveasfilename(initialdir="",title="Save DED Data JSON File as",filetypes=[('JSON files','*.json')])
    root.root.lower()
    if root.savefile:
        if not root.savefile.endswith(".json"):root.savefile+=".json"
        root.entry_2.delete(0,last_index=tk.END)
        root.entry_2.insert(0,root.savefile)
        _=savedata(root,root.entry_2)

def openfile(root):
    """``openfile(root)``.\n
Function to choose a DED data file to load in File Explorer."""
    root.loadfile=ctk.filedialog.askopenfilename(initialdir="",title="Select a DED Data JSON File",filetypes=[('JSON files','*.json')])
    root.root.lower()
    if root.loadfile:
        root.entry.delete(0,last_index=tk.END)
        root.entry.insert(0,root.loadfile)

def saveparameters(root):
    if root.pbar.n<int(root.N_Entry.get()):
        root.pbar.total=root.DEDargs[0]=int(root.N_Entry.get())
        root.N_Entry.configure(state="disabled")
        root.start_button.configure(state="normal")
        root.sidebar_button_3.configure(state="disabled")
        root.pbar.refresh()
    else:
        root.N_Entry.delete(0,last_index=tk.END)
        root.N_Entry.insert(0,str(root.pbar.n))
    root.parainitialized,root.DEDargs[1:],root.progressbar_1.Total,root.progressbar_1.itnum=True,[int(root.scaling_optionemenu.get()),float(root.U_Entry.get()),float(root.Sigma_Entry.get()),float(root.Ed_Entry.get()),float(root.Gamma_Entry.get()),root.ctype_optionemenu.get(),root.Edcalc_optionemenu.get(),int(root.Nimpurities_optionemenu.get()),float(root.U2_Entry.get()),float(root.J_Entry.get()),literal_eval(root.Tk_Entry.get()),literal_eval(root.eta_Entry.get()),int(root.SizeO_Entry.get()),float(root.bound_Entry.get()),root.DEDargs[15],bool(root.log_checkbox.get()),float(root.base_Entry.get()),float(root.graphymax_Entry.get()),bool(root.graphlogy_checkbox.get()),root.graphfDOScolor_Entry.get()],root.DEDargs[0],root.pbar.n
    root.progressbar_1.set(root.pbar.n/root.DEDargs[0])
    root.U_Entry.configure(state="disabled")
    root.Sigma_Entry.configure(state="disabled")
    root.Ed_Entry.configure(state="disabled")
    root.Gamma_Entry.configure(state="disabled")
    root.scaling_optionemenu.configure(state="disabled")
    root.Tk_Entry.configure(state="disabled")
    root.eta_Entry.configure(state="disabled")
    root.SizeO_Entry.configure(state="disabled")
    root.bound_Entry.configure(state="disabled")
    root.ctype_optionemenu.configure(state="disabled")
    root.log_checkbox.configure(state="disabled")
    root.base_Entry.configure(state="disabled")
    root.Edcalc_optionemenu.configure(state="disabled")
    root.Nimpurities_optionemenu.configure(state="disabled")
    root.U2_Entry.configure(state="disabled")
    root.J_Entry.configure(state="disabled")

def paraloader(root):
    root.DEDargs=[root.data["Ntot"],root.data["poles"],root.data["U"],root.data["Sigma"],root.data["Ed"],root.data["Gamma"],root.data["ctype"],root.data["Edcalc"],root.data["Nimpurities"],root.data["U2"],root.data["J"],root.data["Tk"],root.data["etaco"],root.data["SizeO"],root.data["bound"],root.data["posb"],root.data["log"],root.data["base"],root.DEDargs[18],root.DEDargs[19],root.DEDargs[20]]
    root.progressbar_1.Total,root.progressbar_1.itnum=root.pbar.total,root.pbar.n=root.DEDargs[0],root.data["Nit"]
    root.progressbar_1.set(root.pbar.n/root.DEDargs[0])
    root.eta,root.Npoles,root.c,root.Lor=root.DEDargs[12][0]*abs(root.omega)+root.DEDargs[12][1],int(root.DEDargs[1]/root.DEDargs[8]),[Jordan_wigner_transform(i,2*root.DEDargs[1]) for i in range(2*root.DEDargs[1])],Lorentzian(root.omega,root.DEDargs[5],root.DEDargs[1],root.DEDargs[4],root.DEDargs[3])[0]
    (root.Hn,root.n),root.loaded,root.telapsed=Operators(root.c,root.DEDargs[8],root.DEDargs[1]),True,root.data["telapsed"]
    root.pbar.refresh()
    root.U_Entry.delete(0,last_index=tk.END)
    root.U_Entry.insert(0,str(root.DEDargs[2]))
    root.Sigma_Entry.delete(0,last_index=tk.END)
    root.Sigma_Entry.insert(0,str(root.DEDargs[3]))
    root.Ed_Entry.delete(0,last_index=tk.END)
    root.Ed_Entry.insert(0,str(root.DEDargs[4]))
    root.Gamma_Entry.delete(0,last_index=tk.END)
    root.Gamma_Entry.insert(0,str(root.DEDargs[5]))
    root.scaling_optionemenu.set(str(root.DEDargs[1]))
    root.Tk_Entry.delete(0,last_index=tk.END)
    root.Tk_Entry.insert(0,str(root.DEDargs[11]))
    root.eta_Entry.delete(0,last_index=tk.END)
    root.eta_Entry.insert(0,str(root.DEDargs[12]))
    root.SizeO_Entry.delete(0,last_index=tk.END)
    root.SizeO_Entry.insert(0,str(root.DEDargs[13]))
    root.bound_Entry.delete(0,last_index=tk.END)
    root.bound_Entry.insert(0,str(root.DEDargs[14]))
    root.ctype_optionemenu.set(str(root.DEDargs[6]))
    root.log_checkbox.configure(variable=tk.IntVar(value=int(root.DEDargs[16])))
    root.base_Entry.delete(0,last_index=tk.END)
    root.base_Entry.insert(0,str(root.DEDargs[17]))
    root.Edcalc_optionemenu.set(str(root.DEDargs[7]))
    root.Nimpurities_optionemenu.set(str(root.DEDargs[8]))
    root.U2_Entry.delete(0,last_index=tk.END)
    root.U2_Entry.insert(0,str(root.DEDargs[9]))
    root.J_Entry.delete(0,last_index=tk.END)
    root.J_Entry.insert(0,str(root.DEDargs[10]))
    if root.pbar.n>root.DEDargs[0]:
        root.N_Entry.delete(0,last_index=tk.END)
        root.N_Entry.insert(0,str(root.pbar.n))
    else:
        root.N_Entry.delete(0,last_index=tk.END)
        root.N_Entry.insert(0,str(root.DEDargs[0]))   
    if root.pbar.n==root.DEDargs[0]:root.start_button.configure(state="disabled")             
    root.U_Entry.configure(state="disabled")
    root.Sigma_Entry.configure(state="disabled")
    root.Ed_Entry.configure(state="disabled")
    root.Gamma_Entry.configure(state="disabled")
    root.scaling_optionemenu.configure(state="disabled")
    root.Tk_Entry.configure(state="disabled")
    root.eta_Entry.configure(state="disabled")
    root.SizeO_Entry.configure(state="disabled")
    root.bound_Entry.configure(state="disabled")
    root.ctype_optionemenu.configure(state="disabled")
    root.log_checkbox.configure(state="disabled")
    root.base_Entry.configure(state="disabled")
    root.Edcalc_optionemenu.configure(state="disabled")
    root.Nimpurities_optionemenu.configure(state="disabled")
    root.U2_Entry.configure(state="disabled")
    root.J_Entry.configure(state="disabled")

def startDED(root):
    """``startDED(root)``.\n
Function to start DED calculations by initiating loop."""
    if not root.parainitialized:root.parainit()
    if root.parainitialized:
        root.started,root.stopped,root.pbar.start_t,root.autosavsettings=True,False,root.pbar._time()-root.telapsed,{"2":[10000,1],"3":[10000,1],"4":[10000,1],"5":[1000,1],"6":[100,100]}[str(root.DEDargs[1])]
        root.start_button.configure(state="disabled")
        root.pause_button.configure(state="normal")
        root.stop_button.configure(state="normal")
        root.reset_button.configure(state="normal")
        root.main_button.configure(state="disabled")
        loopDED(root)

def loopDED(root):
    """``loopDED(root)``.\n
Function of the main SAIM DED loop which repeats iterations by repeatedly executing ``loopDED(root)``."""
    if not root.stopped and not root.paused and root.pbar.n!=root.DEDargs[0]:
        root.iterationDED()
        root.progressbar_1.itnum=root.pbar.n
        root.progressbar_1.set(root.pbar.n/root.DEDargs[0])
    if not root.stopped and root.pbar.n<root.DEDargs[0]:
        if root.pbar.n%root.autosavsettings[0]==0 and root.pbar.total>root.autosavsettings[0]:
            _=savedata(root,root.entry_2)
            root.showgraph()
        root.after(root.autosavsettings[1],lambda:loopDED(root))
    elif root.pbar.n>=root.DEDargs[0]: 
        root.pbar.close()
        stopDED(root)
        root.showgraph()
        savegraph(root,root.entry_2)
    elif not root.parainitialized:root.stopped=False

def pauseDED(root):
    """``pauseDED(root)``.\n
Function which pauses DED calculations while executing ``loopDED(root)``."""
    if root.started:root.paused=not root.paused

def stopDED(root):
    """``stopDED(root)``.\n
Function which stops DED calculations in ``loopDED(root)`` and saves progress."""
    if savedata(root,root.entry_2) and not root.stopped and root.started:
        root.stopped,root.paused,root.started=True,False,False
        root.pause_button.configure(state="disabled")
        root.stop_button.configure(state="disabled")
        if root.pbar.n!=root.DEDargs[0]:root.start_button.configure(state="normal")
    elif not root.stopped and not root.paused:pauseDED(root)

def graphwindow(root,omega,fDOS,Lor):
    root.fig,root.axis_font=plt.figure(figsize=(9.5,7.6),dpi=50*root._get_window_scaling()),{'fontname':'Calibri','size':'19'}
    mpl.rc('axes',edgecolor='white')
    plt.rc('legend',fontsize=14)
    plt.rc('font',size=19)
    plt.rc('xtick',labelsize=17,color='white')
    plt.rc('ytick',labelsize=17,color='white')
    plt.gca().set_xlim(xmin=-root.DEDargs[14],xmax=root.DEDargs[14])
    if not bool(root.graphlogy_checkbox.get()):
        plt.gca().set_ylim(bottom=0,top=float(root.graphymax_Entry.get()))
        plt.gca().set_xticks(np.linspace(-root.DEDargs[14],root.DEDargs[14],2*int(root.DEDargs[14])+1),minor=False)
    else:
        plt.yscale('log')
        plt.gca().set_ylim(bottom=0.0001,top=float(root.graphymax_Entry.get()))
        plt.gca().set_xticks(np.linspace(-root.DEDargs[14],root.DEDargs[14],int(root.DEDargs[14])+int(root.DEDargs[14])%2+1),minor=False)     
    plt.xlabel("$\\omega$ [-]",**root.axis_font,color='white')
    plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom",rotation=0,labelpad=30,**root.axis_font,color='white')
    plt.plot(omega,Lor,'--r',linewidth=4,label='$\\rho_0$')
    plt.plot(omega,fDOS,root.graphfDOScolor_Entry.get(),label=root.graphlegend_Entry.get())
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    root.fig.set_facecolor("none")
    plt.gca().set_facecolor("#242424")
    root.plot_frame.configure(fg_color="transparent")
    root.canvas=mpl.backends.backend_tkagg.FigureCanvasTkAgg(root.fig,master=root.plot_frame)
    root.canvas.get_tk_widget().config(bg="#242424")
    root.canvas.draw()
    root.canvas.get_tk_widget().grid(row=1,column=1)
    root.plot_frame.grid_rowconfigure((0,2),weight=1)
    root.plot_frame.grid_columnconfigure((0,2),weight=1)
    root.plot_frame.update()
    plt.close()        

def resetDEDwindow(root):
    root.progressbar_1.set(root.pbar.n/root.DEDargs[0])
    root.pbar.refresh()
    root.paused,root.started,root.stopped,root.parainitialized=False,False,True,False
    root.start_button.configure(state="normal")
    root.pause_button.configure(state="disabled")
    root.stop_button.configure(state="disabled")
    root.reset_button.configure(state="disabled")
    root.main_button.configure(state="normal")
    root.N_Entry.configure(state="normal")
    root.sidebar_button_3.configure(state="normal")
    if not root.loaded:
        root.U_Entry.configure(state="normal")
        root.Sigma_Entry.configure(state="normal")
        root.Ed_Entry.configure(state="normal")
        root.Gamma_Entry.configure(state="normal")
        root.scaling_optionemenu.configure(state="normal")
        root.Tk_Entry.configure(state="normal")
        root.eta_Entry.configure(state="normal")
        root.SizeO_Entry.configure(state="normal")
        root.bound_Entry.configure(state="normal")
        root.ctype_optionemenu.configure(state="normal")
        root.log_checkbox.configure(state="normal")
        root.base_Entry.configure(state="normal")
        root.Edcalc_optionemenu.configure(state="normal")
        root.Nimpurities_optionemenu.configure(state="normal")
        root.U2_Entry.configure(state="normal")
        root.J_Entry.configure(state="normal")

class ProgressBar(ctk.CTkProgressBar):
    """``ProgressBar(ctk.CTkProgressBar)``.\n
Custom progressbar with current (and total) number of iterations displayed on the bar."""
    def __init__(self,root,itnum,Total,*args,**kwargs):
        self.itnum,self.Total=itnum,Total
        super().__init__(*args,**kwargs)
        self._canvas.create_text(0,0,text=f"{self.itnum}/{self.Total}",fill="white",font="TkDefaultFont %i"%int(12*root._get_window_scaling()),anchor="c",tags="progress_text")

    def _update_dimensions_event(self,event):
        super()._update_dimensions_event(event)
        self._canvas.coords("progress_text",event.width/2,event.height/2)

    def set(self,val,**kwargs):
        super().set(val,**kwargs)
        self._canvas.itemconfigure("progress_text",text=f"{self.itnum}/{self.Total}")

def CenterWindowToDisplay(Screen:ctk.CTk,width:int,height:int,scale_factor:float=1.0)->str:
    """``CenterWindowToDisplay(Screen:ctk.CTk,width:int,height:int,scale_factor:float=1.0)``.\n
Calculates the necessary coordinates for a ``customTkinter`` window to be displayed in the center of the screen."""
    return f"{width}x{height}+{int((0.5*(Screen.winfo_screenwidth()-width)-7)*scale_factor)}+{int((0.5*(Screen.winfo_screenheight()-height)-31)*scale_factor)}"

def DEDwindow(root,name,app_width=1100,app_height=580):
    root.title(name)
    root.geometry(CenterWindowToDisplay(root,app_width,app_height,root._get_window_scaling()))
    root.after(200,lambda:root.iconbitmap('DEDicon.ico'))
    root.resizable(width=False,height=False)
    root.after(200,root.focus)
    root.grid_columnconfigure(1,weight=1)
    root.grid_columnconfigure((2,3,4),weight=0)
    root.grid_rowconfigure((0,1,2),weight=1)

def parasidebar(root,frame,parainit):
    frame.grid_rowconfigure(16,weight=1)
    root.logo_label=ctk.CTkLabel(frame,text="AIM Parameters",font=ctk.CTkFont(size=20,weight="bold"))
    root.logo_label.grid(row=0,column=0,padx=20,pady=(20,10))
    root.N_label=ctk.CTkLabel(frame,text="No. of iterations DED:",anchor="w")
    root.N_label.grid(row=1,column=0,padx=20,pady=(5,0))
    root.N_Entry=ctk.CTkEntry(frame,placeholder_text="200000")
    root.N_Entry.grid(row=2,column=0,padx=20,pady=(0,0))
    root.U_label=ctk.CTkLabel(frame,text="Coulomb repulsion U:",anchor="w")
    root.U_label.grid(row=3,column=0,padx=20,pady=(5,0))
    root.U_Entry=ctk.CTkEntry(frame,placeholder_text="3")
    root.U_Entry.grid(row=4,column=0,padx=20,pady=(0,0))
    root.Sigma_label=ctk.CTkLabel(frame,text="Effective one-body potential \u03A3\N{SUBSCRIPT ZERO}:",anchor="w")
    root.Sigma_label.grid(row=5,column=0,padx=20,pady=(5,0))
    root.Sigma_Entry=ctk.CTkEntry(frame,placeholder_text="1.5")
    root.Sigma_Entry.grid(row=6,column=0,padx=20,pady=(0,0))
    root.Ed_label=ctk.CTkLabel(frame,text="Electron impurity energy \u03B5d:",anchor="w")
    root.Ed_label.grid(row=7,column=0,padx=20,pady=(5,0))
    root.Ed_Entry=ctk.CTkEntry(frame,placeholder_text="-1.5")
    root.Ed_Entry.grid(row=8,column=0,padx=20,pady=(0,0))
    root.Gamma_label=ctk.CTkLabel(frame,text="Flat hybridization constant \u0393:",anchor="w")
    root.Gamma_label.grid(row=9,column=0,padx=20,pady=(5,0))
    root.Gamma_Entry=ctk.CTkEntry(frame,placeholder_text="0.3")
    root.Gamma_Entry.grid(row=10,column=0,padx=20,pady=(0,0))
    root.scaling_label=ctk.CTkLabel(frame,text="No. of poles:",anchor="w")
    root.scaling_label.grid(row=11,column=0,padx=20,pady=(5,0))
    root.scaling_optionemenu=ctk.CTkOptionMenu(frame,values=["2","3","4","5","6"])
    root.scaling_optionemenu.grid(row=12,column=0,padx=20,pady=(0,0))
    root.Tk_label=ctk.CTkLabel(frame,text="Simulated Temperatures kbT:",anchor="w")
    root.Tk_label.grid(row=13,column=0,padx=20,pady=(5,0))
    root.Tk_Entry=ctk.CTkEntry(frame,placeholder_text="[0]")
    root.Tk_Entry.grid(row=14,column=0,padx=20,pady=(0,0))
    root.sidebar_button_3=ctk.CTkButton(frame,text="Submit Parameters",command=parainit)
    root.sidebar_button_3.grid(row=15,column=0,padx=20,pady=(15,15))
    root.N_Entry.insert(0,str(root.DEDargs[0]))
    root.U_Entry.insert(0,str(root.DEDargs[2]))
    root.Sigma_Entry.insert(0,str(root.DEDargs[3]))
    root.Ed_Entry.insert(0,str(root.DEDargs[4]))
    root.Gamma_Entry.insert(0,str(root.DEDargs[5]))
    root.scaling_optionemenu.set(str(root.DEDargs[1]))
    root.Tk_Entry.insert(0,str(root.DEDargs[11]))

def progressframe(root,frame):
    frame.grid_columnconfigure((0,1,2,3,4,5),weight=1)
    root.progressbar_1=ProgressBar(master=frame,variable=ctk.IntVar(value=0),root=root,itnum=root.pbar.n,Total=root.pbar.total,height=30)
    root.progressbar_1.grid(row=0,column=0,columnspan=6,padx=20,pady=(5,0),sticky="nsew")
    root.start_button=ctk.CTkButton(frame,text="Start",command=lambda:startDED(root))
    root.start_button.grid(row=1,column=0,padx=4,pady=(5,10))
    root.pause_button=ctk.CTkButton(frame,text="Pause",command=lambda:pauseDED(root))
    root.pause_button.grid(row=1,column=1,padx=4,pady=(5,10))
    root.stop_button=ctk.CTkButton(frame,text="Stop",command=lambda:stopDED(root))
    root.stop_button.grid(row=1,column=2,padx=4,pady=(5,10))
    root.show_button=ctk.CTkButton(frame,text="Show Graph",command=root.showgraph)
    root.show_button.grid(row=1,column=3,padx=4,pady=(5,10))
    root.save_button=ctk.CTkButton(frame,text="Save Graph",command=lambda:savegraph(root,root.entry_2))
    root.save_button.grid(row=1,column=4,padx=4,pady=(5,10))
    root.reset_button=ctk.CTkButton(frame,text="Reset",command=root.resetDED)
    root.reset_button.grid(row=1,column=5,padx=4,pady=(5,10))
    root.pause_button.configure(state="disabled")
    root.stop_button.configure(state="disabled")
    root.reset_button.configure(state="disabled")

def settingstab(root,frame):
    frame.add("Adv.")
    frame.add("Multi orb.")
    frame.add("Graph")
    frame.tab("Adv.").grid_columnconfigure(0,weight=1)
    frame.tab("Multi orb.").grid_columnconfigure(0,weight=1)
    frame.tab("Graph").grid_columnconfigure(0,weight=1)
    root.scrollable_tab=ctk.CTkScrollableFrame(frame.tab("Adv."),height=360,width=220,label_text="Advanced settings",label_font=ctk.CTkFont(size=20,weight="bold"),label_fg_color="transparent")
    root.scrollable_tab.grid(row=0, column=0,rowspan=2, padx=(0, 0), pady=(0, 0))
    root.eta_label=ctk.CTkLabel(root.scrollable_tab,text="Imaginary part of frequency arg. of\nGreen's function \u03B7 ([slope,offset]):",anchor="w")
    root.eta_label.grid(row=0,column=0,padx=10,pady=(5,0))
    root.eta_Entry=ctk.CTkEntry(root.scrollable_tab,placeholder_text="[0.02, 1e-39]")
    root.eta_Entry.grid(row=1,column=0,padx=10,pady=(2,0))
    root.eta_Entry.insert(0,str(root.DEDargs[12]))
    root.SizeO_label=ctk.CTkLabel(root.scrollable_tab,text="No. of energies in spectrum \u03C9:",anchor="w")
    root.SizeO_label.grid(row=2,column=0,padx=10,pady=(5,0))
    root.SizeO_Entry=ctk.CTkEntry(root.scrollable_tab,placeholder_text="1001")
    root.SizeO_Entry.grid(row=3,column=0,padx=10,pady=(0,0))
    root.SizeO_Entry.insert(0,str(root.DEDargs[13]))
    root.bound_label=ctk.CTkLabel(root.scrollable_tab,text="Range energies in spectrum \u03C9:",anchor="w")
    root.bound_label.grid(row=4,column=0,padx=10,pady=(5,0))
    root.bound_Entry=ctk.CTkEntry(root.scrollable_tab,placeholder_text="3")
    root.bound_Entry.grid(row=5,column=0,padx=10,pady=(0,0))
    root.bound_Entry.insert(0,str(root.DEDargs[14]))
    root.ctype_label=ctk.CTkLabel(root.scrollable_tab,text="Constraint type:",anchor="w")
    root.ctype_label.grid(row=6,column=0,padx=10,pady=(5,0))
    root.ctype_optionemenu=ctk.CTkOptionMenu(root.scrollable_tab,values=["n"," ","n%2","sn","ssn","dn","mosn"])
    root.ctype_optionemenu.grid(row=7,column=0,padx=10,pady=(0,0))
    root.log_checkbox=ctk.CTkCheckBox(master=root.scrollable_tab,text="Logarithmic scale energies \u03C9")
    root.log_checkbox.grid(row=8,column=0,padx=10,pady=(10,0))
    root.base_label=ctk.CTkLabel(root.scrollable_tab,text="Base log. scale:",anchor="w")
    root.base_label.grid(row=9,column=0,padx=10,pady=(5,0))
    root.base_Entry=ctk.CTkEntry(root.scrollable_tab,placeholder_text="1.5")
    root.base_Entry.grid(row=10,column=0,padx=10,pady=(0,0))
    root.base_Entry.insert(0,str(root.DEDargs[17]))
    root.Edcalc_label=ctk.CTkLabel(root.scrollable_tab,text="\u03B5d in interacting DOS:",anchor="w")
    root.Edcalc_label.grid(row=11,column=0,padx=10,pady=(5,0))
    root.Edcalc_optionemenu=ctk.CTkOptionMenu(root.scrollable_tab,values=["","AS"])
    root.Edcalc_optionemenu.grid(row=12,column=0,padx=10,pady=(0,0))
    root.tab_label2=ctk.CTkLabel(frame.tab("Multi orb."),text="Multi orb. settings",font=ctk.CTkFont(size=20,weight="bold"))
    root.tab_label2.grid(row=0,column=0,padx=10,pady=(10,10))
    root.Nimpurities_label=ctk.CTkLabel(frame.tab("Multi orb."),text="No. of orbitals:",anchor="w")
    root.Nimpurities_label.grid(row=1,column=0,padx=10,pady=(5,0))
    root.Nimpurities_optionemenu=ctk.CTkOptionMenu(frame.tab("Multi orb."),values=["1","2"])
    root.Nimpurities_optionemenu.grid(row=2,column=0,padx=10,pady=(0,0))
    root.U2_label=ctk.CTkLabel(frame.tab("Multi orb."),text="Inter-orbital Coulomb repulsion U':",anchor="w")
    root.U2_label.grid(row=3,column=0,padx=10,pady=(5,0))
    root.U2_Entry=ctk.CTkEntry(frame.tab("Multi orb."),placeholder_text="0")
    root.U2_Entry.grid(row=4,column=0,padx=10,pady=(0,0))
    root.U2_Entry.insert(0,str(root.DEDargs[9]))
    root.J_label=ctk.CTkLabel(frame.tab("Multi orb."),text="Hund’s rule coupling JH:",anchor="w")
    root.J_label.grid(row=5,column=0,padx=10,pady=(5,0))
    root.J_Entry=ctk.CTkEntry(frame.tab("Multi orb."),placeholder_text="0")
    root.J_Entry.grid(row=6,column=0,padx=10,pady=(0,0))
    root.J_Entry.insert(0,str(root.DEDargs[10]))
    root.tab_label3=ctk.CTkLabel(frame.tab("Graph"),text="Graph settings",font=ctk.CTkFont(size=20,weight="bold"))
    root.tab_label3.grid(row=0,column=0,padx=10,pady=(10,10))
    root.graphlegend_label=ctk.CTkLabel(frame.tab("Graph"),text="Legend label:",anchor="w")
    root.graphlegend_label.grid(row=1,column=0,padx=10,pady=(5,0))
    root.graphlegend_Entry=ctk.CTkEntry(frame.tab("Graph"),width=200,placeholder_text='$\\rho_{constr.},N,$n='+str(root.DEDargs[1]))
    root.graphlegend_Entry.grid(row=2,column=0,padx=10,pady=(0,0))
    root.graphlegend_Entry.insert(0,'$\\rho_{constr.},N,$n='+str(root.DEDargs[1]))
    root.graphymax_label=ctk.CTkLabel(frame.tab("Graph"),text="Limit y-axis:",anchor="w")
    root.graphymax_label.grid(row=3,column=0,padx=10,pady=(5,0))
    root.graphymax_Entry=ctk.CTkEntry(frame.tab("Graph"),placeholder_text='1.2')
    root.graphymax_Entry.grid(row=4,column=0,padx=10,pady=(0,0))
    root.graphymax_Entry.insert(0,str(root.DEDargs[18]))
    root.graphlogy_checkbox=ctk.CTkCheckBox(master=frame.tab("Graph"),text="Logarithmic y scale")
    root.graphlogy_checkbox.grid(row=5,column=0,padx=10,pady=(10,0))
    root.graphfDOScolor_label=ctk.CTkLabel(frame.tab("Graph"),text="Interacting DOS color/style:",anchor="w")
    root.graphfDOScolor_label.grid(row=6,column=0,padx=10,pady=(5,0))
    root.graphfDOScolor_Entry=ctk.CTkEntry(frame.tab("Graph"),placeholder_text='b')
    root.graphfDOScolor_Entry.grid(row=7,column=0,padx=10,pady=(0,0))
    root.graphfDOScolor_Entry.insert(0,str(root.DEDargs[20]))

def fileentrytab(root,frame,brow=3,bcol=3):
    frame.grid_columnconfigure((0,1),weight=1)
    frame.grid_columnconfigure((2,3),weight=0)
    root.entry=ctk.CTkEntry(frame,placeholder_text="C:\\")
    root.entry.grid(row=0,column=0,columnspan=2,pady=(10,5),sticky='nsew')
    root.entry_2=ctk.CTkEntry(frame,placeholder_text="example.json")
    root.entry_2.grid(row=1,column=0,columnspan=2,pady=(5,10),sticky='nsew')
    root.file_button_frame=ctk.CTkFrame(root,height=90,fg_color="transparent")
    root.file_button_frame.grid(row=brow,column=bcol,rowspan=2,columnspan=2,padx=(20,20),sticky="nsew")
    root.openfile_button=ctk.CTkButton(root.file_button_frame,text="Open file",fg_color="transparent",width=100,border_width=2,text_color=("gray10","#DCE4EE"),command=lambda:openfile(root))
    root.openfile_button.grid(row=0,column=0,padx=(0,10),pady=(10,5),sticky='ew')
    root.main_button=ctk.CTkButton(root.file_button_frame,text="Submit file to load",fg_color="transparent",width=151,border_width=2,text_color=("gray10","#DCE4EE"),command=root.fileloader)
    root.main_button.grid(row=0,column=1,pady=(10,5),sticky='ew')
    root.saveasfile_button=ctk.CTkButton(root.file_button_frame,text="Save as file",fg_color="transparent",width=100,border_width=2,text_color=("gray10","#DCE4EE"),command=lambda:saveasfile(root))
    root.saveasfile_button.grid(row=1,column=0,padx=(0,10),pady=(5,10),sticky='ew')
    root.main_button_2=ctk.CTkButton(root.file_button_frame,text="Submit file to save",fg_color="transparent",width=151,border_width=2,text_color=("gray10","#DCE4EE"),command=lambda:savfilename(root,root.entry_2))
    root.main_button_2.grid(row=1,column=1,pady=(5,10),sticky='ew')

class mainApp(ctk.CTk):
    """``mainApp(ctk.CTk)``.\n
The main ``customTkinter`` class for the DED Anderson impurity model simulator application window."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.app_width,self.app_height,self.top_level_windows=300,200,{"SAIM single sim":SAIMWINDOW,"Sampled poles Distr.":polesWINDOW,"ASAIM single sim":ASAIMWINDOW,"GNR SAIM single sim":GNRWINDOW,"Impurity Entropy":EntropyWINDOW,"Stdev calculator":StdevWINDOW}
        self.geometry(CenterWindowToDisplay(self,self.app_width,self.app_height,self._get_window_scaling()))
        self.iconbitmap('DEDicon.ico')
        self.resizable(width=False,height=False)
        self.title("DED simulator menu")
        self.focus()
        self.grid_rowconfigure((0,1,2),weight=0)
        self.grid_columnconfigure(0,weight=1)
        self.menu_label=ctk.CTkLabel(self,text="Choose simulation type",font=ctk.CTkFont(size=20,weight="bold"))
        self.menu_label.grid(row=0,column=0,padx=20,pady=(20,10))
        self.scaling_optionemenu=ctk.CTkOptionMenu(self,width=200,values=["SAIM single sim","Sampled poles Distr.","ASAIM single sim","GNR SAIM single sim","Impurity Entropy","Stdev calculator"])
        self.scaling_optionemenu.grid(row=1,column=0,padx=20,pady=(5,0))
        self.button_open=ctk.CTkButton(self,text="Open",command=lambda:self.open_toplevel(simsel=self.scaling_optionemenu.get()))
        self.button_open.grid(row=2,column=0,padx=20,pady=(20,20))
        self.protocol('WM_DELETE_WINDOW',self.quitApp)

    def open_toplevel(self,simsel):
        """``open_toplevel(self,simsel)``.\n
    Class method to initialize selected window for specific simulation type."""
        self.toplevel_window=self.top_level_windows[simsel](selfroot=self)
        self.after(100,self.lower)
        self.scaling_optionemenu.configure(state="disabled")
        self.button_open.configure(state="disabled")

    def quitApp(self):
        """``quitApp(self)``.\n
    Class method to show messagebox when attempting to close main window."""
        self.msg=CTkMessagebox(master=self,title="Exit?",message="Are you sure you want to quit the entire application?",icon="question",option_1="Cancel",option_2="No",option_3="Yes")
        if self.msg.get()=="Yes":self.destroy()

class SAIMWINDOW(ctk.CTkToplevel):
    """``SAIMWINDOW(ctk.CTkToplevel)``.\n
Class for Symmetric Anderson impurity model DED simmulation window."""
    def __init__(self,selfroot,N=200000,poles=4,U=3,Sigma=3/2,Ed=-3/2,Gamma=0.3,SizeO=1001,etaco=[0.02,1e-39],ctype='n',Edcalc='',bound=3,Tk=[0],Nimpurities=1,U2=0,J=0,posb=1,log=False,base=1.5,ymax=1.2,logy=False,fDOScolor='b',*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.simtype,self.root,self.paused,self.started,self.stopped,self.loaded,self.parainitialized,self.poleDOS,self.telapsed,self.DEDargs="SAIM",selfroot,False,False,False,False,False,False,0,[N,poles,U,Sigma,Ed,Gamma,ctype,Edcalc,Nimpurities,U2,J,Tk,etaco,SizeO,bound,posb,log,base,ymax,logy,fDOScolor]
        if self.DEDargs[16]:self.omega,self.Npoles=np.concatenate((-np.logspace(np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),np.log(1e-5)/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17]),np.logspace(np.log(1e-5)/np.log(self.DEDargs[17]),np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17]))),int(self.DEDargs[1]/self.DEDargs[8])
        else:self.omega,self.Npoles=np.linspace(-self.DEDargs[14],self.DEDargs[14],self.DEDargs[13]),int(self.DEDargs[1]/self.DEDargs[8])
        self.c,self.pbar,self.eta=[Jordan_wigner_transform(i,2*self.DEDargs[1]) for i in range(2*self.DEDargs[1])],trange(self.DEDargs[0],position=self.DEDargs[15],leave=False,desc='Iterations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),self.DEDargs[12][0]*abs(self.omega)+self.DEDargs[12][1]
        (self.Hn,self.n),self.AvgSigmadat,self.Nfin,self.nd,self.Lor=Operators(self.c,self.DEDargs[8],self.DEDargs[1]),np.zeros((len(self.DEDargs[11]),self.DEDargs[13]),dtype='complex_'),np.zeros(len(self.DEDargs[11]),dtype='float'),np.zeros(len(self.DEDargs[11]),dtype='complex_'),Lorentzian(self.omega,self.DEDargs[5],self.DEDargs[1],self.DEDargs[4],self.DEDargs[3])[0]
        DEDwindow(self,"Distributional Exact Diagonalization AIM simulator")
        self.sidebar_frame=ctk.CTkFrame(self,width=140,corner_radius=0)
        self.sidebar_frame.grid(row=0,column=0,rowspan=5,sticky="nsew")
        parasidebar(self,self.sidebar_frame,self.parainit)
        self.plot_frame=ctk.CTkFrame(self,width=250,height=380)
        self.plot_frame.grid(row=0,column=1,columnspan=2,rowspan=1,padx=(20,0),pady=(20,10),sticky="nsew")
        self.slider_progressbar_frame=ctk.CTkFrame(self)
        self.slider_progressbar_frame.grid(row=1,column=1,columnspan=2,padx=(20,0),pady=(5,0),sticky="nsew")
        progressframe(self,self.slider_progressbar_frame)
        self.file_entry_frame=ctk.CTkFrame(self,height=90,fg_color="transparent")
        self.file_entry_frame.grid(row=3,column=1,rowspan=2,columnspan=2,padx=(20,0),sticky="nsew")
        fileentrytab(self,self.file_entry_frame)
        self.settings_tab=ctk.CTkTabview(self, width=261)
        self.settings_tab.grid(row=0,column=3,rowspan=2,columnspan=2,padx=(20,20),pady=(20,0),sticky="nsew")
        settingstab(self,self.settings_tab)
        self.protocol('WM_DELETE_WINDOW',lambda:enableroot(self))

    def parainit(self):
        """``parainit(self)``.\n
    Class method to save input parameters of DED from inputs in the ``SAIMWINDOW(ctk.CTkToplevel)`` window."""
        try:
            saveparameters(self)
            if not self.loaded:
                if self.DEDargs[16]:self.omega=np.concatenate((-np.logspace(np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),np.log(1e-5)/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17]),np.logspace(np.log(1e-5)/np.log(self.DEDargs[17]),np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17])))
                else:self.omega=np.linspace(-self.DEDargs[14],self.DEDargs[14],self.DEDargs[13])
                self.eta,self.AvgSigmadat,self.Nfin,self.nd,self.Npoles,self.c,self.Lor=self.DEDargs[12][0]*abs(self.omega)+self.DEDargs[12][1],np.zeros((len(self.DEDargs[11]),self.DEDargs[13]),dtype='complex_'),np.zeros(len(self.DEDargs[11]),dtype='float'),np.zeros(len(self.DEDargs[11]),dtype='complex_'),int(self.DEDargs[1]/self.DEDargs[8]),[Jordan_wigner_transform(i,2*self.DEDargs[1]) for i in range(2*self.DEDargs[1])],Lorentzian(self.omega,self.DEDargs[5],self.DEDargs[1],self.DEDargs[4],self.DEDargs[3])[0]
                (self.Hn,self.n)=Operators(self.c,self.DEDargs[8],self.DEDargs[1])
        except:pass

    def fileloader(self):
        """``fileloader(self)``.\n
    Class method to load ``.json`` file and save data and settings from that particular simulation session to utilize for current session."""
        try:
            self.data=json.load(open(self.entry.get()))
            if self.data["simtype"]==self.simtype:
                self.Nfin,self.omega,self.AvgSigmadat,self.nd=np.array(self.data["Nfin"]),np.array(self.data["omega"]),(np.array(self.data["AvgSigmadat"],dtype='complex_')*np.array(self.data["Nfin"])).squeeze(),np.array(self.data["nd"],dtype='complex_')*np.array(self.data["Nfin"])
                paraloader(self)
            else:
                self.entry.delete(0,last_index=tk.END)
                self.entry.insert(0,'Try again')
        except IOError or FileNotFoundError:
            self.entry.delete(0,last_index=tk.END)
            self.entry.insert(0,'Try again')

    def iterationDED(self,reset=False):
        """``iterationDED(self,reset=False)``.\n
    Class method of a single DED iteration which updates ``self.AvgSigmadat`` and other results for each iteration."""
        while not reset:
            self.NewM,self.nonG,_=Startrans(self.Npoles,np.sort(Lorentzian(self.omega,self.DEDargs[5],self.Npoles,self.DEDargs[4],self.DEDargs[3])[1]),self.omega,self.eta)
            self.H0,self.H=HamiltonianAIM(np.repeat(self.NewM[0][0],self.DEDargs[8]),np.tile([self.NewM[k+1][k+1] for k in range(len(self.NewM)-1)],(self.DEDargs[8],1)),np.tile(self.NewM[0,1:],(self.DEDargs[8],1)),self.DEDargs[2],self.DEDargs[3],self.DEDargs[9],self.DEDargs[10],self.Hn)
            try:(self.MBGdat,self.Boltzmann,self.Ev0),reset=Constraint(self.DEDargs[6],self.H0,self.H,self.omega,self.eta,self.c,self.n,self.DEDargs[11],np.array([ar<self.DEDargs[0] for ar in self.Nfin]),self.poleDOS)
            except (np.linalg.LinAlgError,ValueError,scipy.sparse.linalg.ArpackNoConvergence):(self.MBGdat,self.Boltzmann,self.Ev0),reset=(np.zeros(len(self.omega),dtype='complex_'),np.zeros(len(self.DEDargs[11])),np.array([])),False
            if np.isnan(1/self.nonG-1/self.MBGdat+self.DEDargs[3]).any() or np.array([i>=1000 for i in np.real(1/self.nonG-1/self.MBGdat+self.DEDargs[3])]).any():reset=False
        self.Nfin,self.AvgSigmadat,self.nd=self.Nfin+self.Boltzmann,self.AvgSigmadat+(1/self.nonG-1/self.MBGdat+self.DEDargs[3])*self.Boltzmann[:,None],self.nd+np.conj(self.Ev0).T@sum(self.Hn[0]).data.tocoo()@self.Ev0*self.Boltzmann
        if self.DEDargs[6]=='sn':self.pbar.n+=1
        else:self.pbar.n=int(min(self.Nfin))
        self.pbar.refresh()

    def showgraph(self):
        """``showgraph(self)``.\n
    Class method to show graph of current results based on finished iterations."""
        try:
            if self.DEDargs[7]=='AS':self.fDOS=(-np.imag(np.nan_to_num(1/(self.omega-self.AvgSigmadat/self.Nfin[:,None]+(self.AvgSigmadat[:,int(np.round(self.DEDargs[13]/2))]/self.Nfin)[:,None]+1j*self.DEDargs[5])))/np.pi).squeeze()
            else:self.fDOS=(-np.imag(np.nan_to_num(1/(self.omega-self.AvgSigmadat/self.Nfin[:,None]-self.DEDargs[4]+1j*self.DEDargs[5])))/np.pi).squeeze()
            graphwindow(self,self.omega,self.fDOS,self.Lor)
        except:pass

    def resetDED(self):
        """``resetDED(self)``.\n
    Class method to reset DED calculation progress in ``loopDED(root)``."""
        self.pbar.reset()
        if self.loaded:
            self.progressbar_1.itnum=self.pbar.n=self.data["Nit"]
            self.Nfin,self.AvgSigmadat,self.nd,self.telapsed=np.array(self.data["Nfin"]),(np.array(self.data["AvgSigmadat"],dtype='complex_')*np.array(self.data["Nfin"])).squeeze(),np.array(self.data["nd"],dtype='complex_')*np.array(self.data["Nfin"]),self.data["telapsed"]
        else: 
            self.progressbar_1.itnum=self.pbar.n=0
            self.AvgSigmadat,self.Nfin,self.nd,self.telapsed=np.zeros((len(self.DEDargs[11]),self.DEDargs[13]),dtype='complex_'),np.zeros(len(self.DEDargs[11]),dtype='float'),np.zeros(len(self.DEDargs[11]),dtype='complex_'),0
        resetDEDwindow(self)

class polesWINDOW(ctk.CTkToplevel):
    """``polesWINDOW(ctk.CTkToplevel)``.\n
Class for sampled poles distribution calculator DED simmulation window."""
    def __init__(self,selfroot,N=200000,poles=4,U=3,Sigma=3/2,Ed=-3/2,Gamma=0.3,SizeO=1001,etaco=[0.02,1e-39],ctype='n',Edcalc='',bound=3,Tk=[0],Nimpurities=1,U2=0,J=0,posb=1,log=False,base=1.5,ymax=1.2,logy=False,fDOScolor='b',*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.simtype,self.root,self.paused,self.started,self.stopped,self.loaded,self.parainitialized,self.poleDOS,self.ratio,self.corrfactor,self.telapsed,self.DEDargs="poles",selfroot,False,False,False,False,False,True,200,1,0,[N,poles,U,Sigma,Ed,Gamma,ctype,Edcalc,Nimpurities,U2,J,Tk,etaco,SizeO,bound,posb,log,base,ymax,logy,fDOScolor]
        if self.DEDargs[16]:self.omega,self.Npoles=np.concatenate((-np.logspace(np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),np.log(1e-5)/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17]),np.logspace(np.log(1e-5)/np.log(self.DEDargs[17]),np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17]))),int(self.DEDargs[1]/self.DEDargs[8])
        else:self.omega,self.Npoles=np.linspace(-self.DEDargs[14],self.DEDargs[14],self.DEDargs[13]),int(self.DEDargs[1]/self.DEDargs[8])
        self.omegaintrv,self.c,self.pbar,self.eta=np.linspace(-self.DEDargs[14],self.DEDargs[14],int(self.DEDargs[0]*self.DEDargs[1]/self.ratio)),[Jordan_wigner_transform(i,2*self.DEDargs[1]) for i in range(2*self.DEDargs[1])],trange(self.DEDargs[0],position=self.DEDargs[15],leave=False,desc='Iterations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),self.DEDargs[12][0]*abs(self.omega)+self.DEDargs[12][1]
        self.omegap,self.DOSp,(self.Hn,self.n),self.Nfin,self.Lor=np.array([(self.omegaintrv[i]+self.omegaintrv[i+1])/2 for i in range(len(self.omegaintrv)-1)]),np.zeros(len(self.omegaintrv)-1,dtype='int_'),Operators(self.c,self.DEDargs[8],self.DEDargs[1]),np.zeros(len(self.DEDargs[11]),dtype='float'),Lorentzian(self.omega,self.DEDargs[5],self.DEDargs[1],self.DEDargs[4],self.DEDargs[3])[0]
        DEDwindow(self,"Distributional Exact Diagonalization AIM sampled poles distribution calculator")
        self.sidebar_frame=ctk.CTkFrame(self,width=140,corner_radius=0)
        self.sidebar_frame.grid(row=0,column=0,rowspan=5,sticky="nsew")
        parasidebar(self,self.sidebar_frame,self.parainit)
        self.plot_frame=ctk.CTkFrame(self,width=250,height=380)
        self.plot_frame.grid(row=0,column=1,columnspan=2,rowspan=1,padx=(20,0),pady=(20,10),sticky="nsew")
        self.slider_progressbar_frame=ctk.CTkFrame(self)
        self.slider_progressbar_frame.grid(row=1,column=1,columnspan=2,padx=(20,0),pady=(5,0),sticky="nsew")
        progressframe(self,self.slider_progressbar_frame)
        self.file_entry_frame=ctk.CTkFrame(self,height=90,fg_color="transparent")
        self.file_entry_frame.grid(row=3,column=1,rowspan=2,columnspan=2,padx=(20,0),sticky="nsew")
        fileentrytab(self,self.file_entry_frame)
        self.settings_tab=ctk.CTkTabview(self, width=261)
        self.settings_tab.grid(row=0,column=3,rowspan=2,columnspan=2,padx=(20,20),pady=(20,0),sticky="nsew")
        settingstab(self,self.settings_tab)
        self.polesratio_label=ctk.CTkLabel(self.scrollable_tab,text="No. of poles per\nenergy interval \u0394\u03C9:",anchor="w")
        self.polesratio_label.grid(row=13,column=0,padx=10,pady=(5,0))
        self.polesratio_Entry=ctk.CTkEntry(self.scrollable_tab,placeholder_text='200')
        self.polesratio_Entry.grid(row=14,column=0,padx=10,pady=(2,0))
        self.polesratio_Entry.insert(0,str(self.ratio))
        self.graphcorrfactor_label=ctk.CTkLabel(self.settings_tab.tab("Graph"),text="Correction factor pole Distr.:",anchor="w")
        self.graphcorrfactor_label.grid(row=8,column=0,padx=10,pady=(5,0))
        self.graphcorrfactor_Entry=ctk.CTkEntry(self.settings_tab.tab("Graph"),placeholder_text='1')
        self.graphcorrfactor_Entry.grid(row=9,column=0,padx=10,pady=(0,0))
        self.graphcorrfactor_Entry.insert(0,str(self.corrfactor))
        self.protocol('WM_DELETE_WINDOW',lambda:enableroot(self))

    def parainit(self):
        """``parainit(self)``.\n
    Class method to save input parameters of DED from inputs in the ``polesWINDOW(ctk.CTkToplevel)`` window."""
        try:
            saveparameters(self)
            self.polesratio_Entry.configure(state="disabled")
            if not self.loaded:
                if self.DEDargs[16]:self.ratio,self.omega=int(self.polesratio_Entry.get()),np.concatenate((-np.logspace(np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),np.log(1e-5)/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17]),np.logspace(np.log(1e-5)/np.log(self.DEDargs[17]),np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17])))
                else:self.ratio,self.omega=int(self.polesratio_Entry.get()),np.linspace(-self.DEDargs[14],self.DEDargs[14],self.DEDargs[13])
                self.omegaintrv,self.eta,self.Nfin,self.Npoles,self.c,self.Lor=np.linspace(-self.DEDargs[14],self.DEDargs[14],int(self.DEDargs[0]*self.DEDargs[1]/self.ratio)),self.DEDargs[12][0]*abs(self.omega)+self.DEDargs[12][1],np.zeros(len(self.DEDargs[11]),dtype='float'),int(self.DEDargs[1]/self.DEDargs[8]),[Jordan_wigner_transform(i,2*self.DEDargs[1]) for i in range(2*self.DEDargs[1])],Lorentzian(self.omega,self.DEDargs[5],self.DEDargs[1],self.DEDargs[4],self.DEDargs[3])[0]
                self.omegap,self.DOSp,(self.Hn,self.n)=np.array([(self.omegaintrv[i]+self.omegaintrv[i+1])/2 for i in range(len(self.omegaintrv)-1)]),np.zeros(len(self.omegaintrv)-1,dtype='int_'),Operators(self.c,self.DEDargs[8],self.DEDargs[1])
        except:pass

    def fileloader(self):
        """``fileloader(self)``.\n
    Class method to load ``.json`` file and save data and settings from that particular simulation session to utilize for current session."""
        try:
            self.data=json.load(open(self.entry.get()))
            if self.data["simtype"]==self.simtype:
                self.Nfin,self.omega,self.omegap,self.DOSp,self.ratio,self.corrfactor=np.array(self.data["Nfin"]),np.array(self.data["omega"]),np.array(self.data["omegap"]),np.array(self.data["DOSp"]),self.data["ratio"],self.data["corrfactor"]
                paraloader(self)
                self.omegaintrv=np.linspace(-self.DEDargs[14],self.DEDargs[14],int(self.DEDargs[0]*self.DEDargs[1]/self.ratio))
                self.polesratio_Entry.delete(0,last_index=tk.END)
                self.polesratio_Entry.insert(0,str(self.ratio))
                self.graphcorrfactor_Entry.delete(0,last_index=tk.END)
                self.graphcorrfactor_Entry.insert(0,str(self.corrfactor))
                self.polesratio_Entry.configure(state="disabled")
            else:
                self.entry.delete(0,last_index=tk.END)
                self.entry.insert(0,'Try again')             
        except IOError or FileNotFoundError:
            self.entry.delete(0,last_index=tk.END)
            self.entry.insert(0,'Try again')

    def showgraph(self):
        """``showgraph(self)``.\n
    Class method to show graph of current results based on finished iterations."""
        try:
            self.fDOS,self.Lorp=polesfDOS(self.DOSp,self.DEDargs[14],float(self.graphcorrfactor_Entry.get())),Lorentzian(self.omegap,self.DEDargs[5],self.DEDargs[1],self.DEDargs[4],self.DEDargs[3])[0]
            graphwindow(self,self.omegap,self.fDOS,self.Lorp)
        except:pass

    def resetDED(self):
        """``resetDED(self)``.\n
    Class method to reset DED calculation progress in ``loopDED(root)``."""
        self.pbar.reset()
        if self.loaded:
            self.progressbar_1.itnum=self.pbar.n=self.data["Nit"]
            self.Nfin,self.DOSp,self.corrfactor,self.telapsed=np.array(self.data["Nfin"]),np.array(self.data["DOSp"]),self.data["corrfactor"],self.data["telapsed"]
            self.graphcorrfactor_Entry.delete(0,last_index=tk.END)
            self.graphcorrfactor_Entry.insert(0,str(self.corrfactor))
        else: 
            self.progressbar_1.itnum=self.pbar.n=0
            self.Nfin,self.DOSp,self.telapsed=np.zeros(len(self.DEDargs[11]),dtype='float'),np.zeros(len(self.omegaintrv)-1,dtype='int_'),0
            self.polesratio_Entry.configure(state="normal")
        resetDEDwindow(self)

    def iterationDED(self,reset=False):
        while not reset:
            self.NewM,self.nonG,self.select=Startrans(self.Npoles,np.sort(Lorentzian(self.omega,self.DEDargs[5],self.Npoles,self.DEDargs[4],self.DEDargs[3])[1]),self.omega,self.eta)
            self.H0,self.H=HamiltonianAIM(np.repeat(self.NewM[0][0],self.DEDargs[8]),np.tile([self.NewM[k+1][k+1] for k in range(len(self.NewM)-1)],(self.DEDargs[8],1)),np.tile(self.NewM[0,1:],(self.DEDargs[8],1)),self.DEDargs[2],self.DEDargs[3],self.DEDargs[9],self.DEDargs[10],self.Hn)
            try:(_,self.Boltzmann,_),reset=Constraint(self.DEDargs[6],self.H0,self.H,self.omega,self.eta,self.c,self.n,self.DEDargs[11],np.array([ar<self.DEDargs[0] for ar in self.Nfin]),self.poleDOS)
            except (np.linalg.LinAlgError,ValueError,scipy.sparse.linalg.ArpackNoConvergence): pass
        self.Nfin,self.DOSp=self.Nfin+self.Boltzmann,self.DOSp+[((self.omegaintrv[i]<=self.select)&(self.select<self.omegaintrv[i+1])).sum() for i in range(0,len(self.omegaintrv)-1)]
        if self.DEDargs[6]=='sn':self.pbar.n+=1
        else:self.pbar.n=int(min(self.Nfin))
        self.pbar.refresh()

class ASAIMWINDOW(ctk.CTkToplevel):
    pass

class GNRWINDOW(ctk.CTkToplevel):
    pass

class EntropyWINDOW(ctk.CTkToplevel):
    pass

class StdevWINDOW(ctk.CTkToplevel):
    pass

if __name__ == "__main__":
    mainApp().mainloop()