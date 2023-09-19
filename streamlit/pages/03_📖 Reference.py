from utilities import *

st.title('Reference')

col1, col2, col3 = st.columns(3)

with col1:

    st.header('Map Visualisation')

    with st.expander("Causal loop diagram"):
        markdown = read_markdown_file("pages/reference/CLD.md")
        st.markdown(markdown, unsafe_allow_html=True)

    with st.expander("Force-directed graph"):
        markdown = read_markdown_file("pages/reference/Force-directed.md")
        st.markdown(markdown, unsafe_allow_html=True)

    with st.expander("Chord diagram"):
        markdown = read_markdown_file("pages/reference/Chord.md")
        st.markdown(markdown, unsafe_allow_html=True)







        

