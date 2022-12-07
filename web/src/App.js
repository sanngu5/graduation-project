import './App.css';
import React, { useState } from 'react';
import FileData from './components/FileData';
import ActionButton from './components/ActionButton';
import {Button, Switch, TextField} from '@mui/material'

const axios = require('axios').default;
axios.defaults.withCredentials = true;

function App() {

  const [youtube, setYouTube] = useState(null);
  const [link, setLink] = useState(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [videoStatus, setVideoStatus] = useState(null);
  const [result, setResult] = useState(null);

  // 파일 선택 버튼
  const onFileChange = (e) => {
    const file = e.target.files[0]
    if(file.type === 'video/mp4') {
      setSelectedFile(file);
      setVideoStatus("UPLOADABLE");
    } else {
      alert('mp4파일만 변환 가능합니다.')
      setSelectedFile(null)
      setVideoStatus("NOVIDEO");
    }
  }

  //mp4 파일 서버로 전송
  const onFileUpload = async () => {
    setVideoStatus("LOADING");
    const formData = new FormData();
    formData.append('send', selectedFile);  
    const res = await axios({
      method: 'post',
      url: 'http://127.0.0.1:5000/upload',
      data: formData,
      headers: {'Content-Type': 'multipart/form-data'}
    })
    .then(function (res){
      // res.
      setVideoStatus("DOWNLOADABLE");
      console.log(res);
      setResult(res.data);
      setSelectedFile(null);
    })
    .catch(function (error){
      setVideoStatus(null)
      setResult(null)
      setSelectedFile(null)
      alert('변환에 실패했습니다. 다시 한 번 시도해주세요.')
    })
  }

  //파일변환 완료에 대한 임시 버튼
  const onFileChanged = () => {
    setSelectedFile(null);
    setVideoStatus("DOWNLOADABLE");
  }

  //변환된 파일 다운로드 코드
  const onFileDownload = () => {
    setSelectedFile(null);
  }

  const onYoutubeChange = () => {
    setYouTube(!youtube)
  }

  return (
    <div 
      className='container' 
      style={{
        "display": "flex",
        "flex-direction": "column",
        "align-items": "center",
        "justify-content": "center",
        "margin": 100
      }}
    >
      <FileData 
        selectedFile={selectedFile}
        videoStatus={videoStatus} 
      />
      <div style={{
        "display": "flex",
        "flex-direction": "row",
        "align-items": "center",
        "justify-content": "center",
        "border": "solid",
        "borderRadius": 10,
        "borderColor": "#E2E2E2",
        "padding": 5
      }}>
        {/* {youtube
        ?
          <>
            <TextField 
              id="youtubelink"
              label="유튜브 링크를 붙여넣으세요"
              variant="standard"
              style={{"height": 50, "minWidth": 400, "margin":5, "color": "gray"}}
            />
            <ActionButton></ActionButton>
          </>
        :<> */}
          <Button
            variant="contained"
            component="label"
          >
            파일 선택
            <input
              type="file"
              hidden
              onChange = {onFileChange}
            />
          </Button>
          <div id="file_name" style={{"minWidth": 300, "margin":5, "color": "gray"}}>
            {selectedFile ? selectedFile.name : "파일을 선택해주세요"}
          </div>
          <ActionButton 
            selectedFile={selectedFile}
            videoStatus={videoStatus}
            onFileUpload={onFileUpload}
            onFileChanged={onFileChanged}
            onFileDownload={onFileDownload}
          />
        {/* </>}
        <Switch
          checked={youtube}
          onChange={onYoutubeChange}
        /> */}
      </div>
    </div>
  )
}

export default App;