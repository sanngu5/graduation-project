import React from "react";
import {Button, Link} from "@mui/material"
import CircularProgress from '@mui/material/CircularProgress';


const ActionButton = (props) => {
  const {
    selectedFile,
    videoStatus,
    onFileUpload,
    onFileChanged,
    onFileDownload,
  } = props;

  if (videoStatus === "LOADING") {
    return (
      <Button disabled={true} onClick={onFileChanged}>
        <CircularProgress size={22} />
      </Button>
    );
  } else if (videoStatus === "DOWNLOADABLE") {
    return (
      <Link href="http://127.0.0.1:5000/download" underline="none" style={{fontSize: 14, paddingRight: 13}}>
        미리보기
      </Link>
    );
  } else {
    return (
      <Button disabled={!selectedFile} onClick={onFileUpload}>
        동영상 변환
      </Button>
    );
  }
};

export default ActionButton;
