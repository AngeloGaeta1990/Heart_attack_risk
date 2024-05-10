import streamlit as st
import pandas as pd
import numpy as np
import joblib

def load_heart_data():
    df = pd.read_csv("outputs/datasets/collection/heart.csv")
    return df


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)