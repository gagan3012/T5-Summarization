import streamlit as st
import yaml

from models import predict_model


def visualize():
    st.write('# Summarization  UI')
    st.markdown(
        '''
        *For additional questions and inquiries, please contact **Gagan Bhatia** via [LinkedIn](
        https://www.linkedin.com/in/gbhatia30/) or [Github](https://github.com/gagan3012).*
        ''')

    text = st.text_area("Enter text here")
    if st.button("Generate Summary"):
        with st.spinner("Connecting the Dots..."):
            sumtext = predict_model(text=text)
        st.write("# Generated Summary:")
        st.write("{}".format(sumtext))
        with open("reports/visualization_metrics.txt", "w") as file1:
            file1.writelines(text)
            file1.writelines(sumtext)


if __name__ == "__main__":
    with open("params.yml") as f:
        params = yaml.safe_load(f)
