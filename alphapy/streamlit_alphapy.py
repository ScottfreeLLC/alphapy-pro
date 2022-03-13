################################################################################
#
# Package   : AlphaPy
# Module    : streamlit
# Created   : February 21, 2021
#
# streamlit run streamlit.py
#
# Copyright 2021 ScottFree Analytics LLC
# Mark Conway & Robert D. Scott II
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import streamlit as st

def app():
    st.header("AlphaPy AutoML")

    # Projects
    """
    home_directory = str(Path.home())
    st.sidebar.subheader("Project Root Directory")

    if st.sidebar.checkbox("Use Home Directory", True):
        root_directory = home_directory
        st.sidebar.markdown(home_directory)
    else:
        root_directory = st.sidebar.text_input('Root Directory', home_directory)
        st.sidebar.markdown(root_directory)

    @st.cache
    def get_projects(file_name, directory):
        paths = []
        for path in Path(directory).rglob(file_name):
            paths.append(path)
        return paths

    st.sidebar.subheader("Projects")

    st.sidebar.markdown(get_projects('model.yml', root_directory))
    """
