import streamlit as st
from src.models.predict_model import predict_model


def visualize():
    st.write("# Summarization  UI")
    st.markdown(
        """
        *For additional questions and inquiries, please contact **Gagan Bhatia** via [LinkedIn](
        https://www.linkedin.com/in/gbhatia30/) or [Github](https://github.com/gagan3012).*
        """
    )

    text = st.text_area("Enter text here")
    if st.button("Generate Summary"):
        with st.spinner("Connecting the Dots..."):
            sumtext = predict_model(text=text)
        st.write("# Generated Summary:")
        st.write("{}".format(sumtext))


if __name__ == "__main__":
    with open("params.yml") as f:
