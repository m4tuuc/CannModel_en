#Testea el rendimiento de la aplicación de Streamlit


import cProfile
import pstats
import io
import streamlit as st

def profile_function(func):
    """Decorator to profile a function."""
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return result
    return wrapper

@profile_function
def main():
    # Tu código de Streamlit aquí
    st.title("Mi Aplicación de Streamlit")
    # ... más código de Streamlit ...

if __name__ == "__main__":
    main()