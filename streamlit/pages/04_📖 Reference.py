from utilities import *

st.title('Reference')

col1, col2 = st.columns([0.30, 0.70])

with col1:

    st.markdown('## :red[*What is the system?*]')

    st.subheader(':blue[Map the system]', divider="blue")

    if st.checkbox("Causal loop diagram"):
        with col2:
            markdown = read_markdown_file("pages/reference/CLD.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()

    if st.checkbox("Influence diagram (Force-Directed Graph)"):
        with col2:
            markdown = read_markdown_file("pages/reference/Force-directed.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()

    if st.checkbox("Chord diagram"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()

    st.subheader(':blue[Reduce Complexity]', divider="blue")

    if st.checkbox("Submaps"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()
    
    st.subheader(":blue[Identify clusters]", divider="blue")

    if st.checkbox("Dendograms (Hierarchical Clustering)"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()


    st.divider()

    st.markdown('## :red[*What are the main elements or drivers of the system?*]')

    st.subheader(":blue[Node importance: Centrality measures]", divider="blue")

    if st.checkbox("Degree centrality"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()

    if st.checkbox("In-degree centrality"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()

    if st.checkbox("Out-degree centrality"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()

    if st.checkbox("Closeness centrality"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()

    if st.checkbox("Betweenness centrality"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()

    if st.checkbox("Pagerank centrality"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()

    st.subheader(":blue[Centrality clustermaps]", divider="blue")

    if st.checkbox("Clustermaps"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()

    st.subheader(":blue[Centrality archetypes]", divider="blue")


    if st.checkbox("Radar plots"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()

    st.divider()

    st.markdown('## :red[*What are the most strategic ways to intervene within a system?*]')

    st.subheader(":blue[Controllability]", divider="blue")

    if st.checkbox("Control Centrality"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()

    if st.checkbox("Robust controllability"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()
    
    if st.checkbox("Global controllability"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()

    st.subheader(":blue[Path Analysis]", divider="blue")

    if st.checkbox("Intended and Unintended Consequences"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()

    st.subheader(":blue[Tradeoff Analysis]", divider="blue")

    if st.checkbox("Interactive Parallel Plots"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()

    st.subheader(":blue[Scenario Analysis]", divider="blue")

    if st.checkbox("Diffusion Models (Pulse and Flow)"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()

    st.subheader(":blue[Optimisation Analysis]", divider="blue")

    if st.checkbox("Genetic Algorithm"):
        with col2:
            markdown = read_markdown_file("pages/reference/Chord.md")
            st.markdown(markdown, unsafe_allow_html=True)
            st.divider()



st.sidebar.markdown('#')
st.sidebar.markdown('#')
st.sidebar.markdown('#')
st.sidebar.markdown('#')
st.sidebar.markdown('#')
st.sidebar.markdown('#')
st.sidebar.markdown('#')
st.sidebar.markdown('#')
st.sidebar.markdown('#')
st.sidebar.markdown('#')
st.sidebar.markdown('#')
st.sidebar.markdown('#')
st.sidebar.markdown('#')
st.sidebar.markdown('#')


st.sidebar.image(str(main_path.joinpath('pysm.png')),  use_column_width='always')



