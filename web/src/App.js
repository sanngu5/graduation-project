import './App.css';
import React, { useState } from 'react';
import FileData from './components/FileData';
import ActionButton from './components/ActionButton';
import STATUS from './constants/Status';

const axios = require('axios').default;
axios.defaults.withCredentials = true;

function App() {

  const [selectedFile, setSelectedFile] = useState(null);
  const [videoStatus, setVideoStatus] = useState(STATUS[0]);
  const [result, setResult] = useState(null);

  // 파일 선택 버튼
  const onFileChange = (e) => {
    const file = e.target.files[0]
    if(file.type === 'video/mp4') {
      setSelectedFile(file);
      setVideoStatus(STATUS["UPLOADABLE"]);
    } else {
      alert('mp4파일만 변환 가능합니다.')
      setSelectedFile(null)
      setVideoStatus(STATUS["NOVIDEO"]);
    }
  }

  //mp4 파일 서버로 전송
  const onFileUpload = async () => {
    const formData = new FormData();
    formData.append('send', selectedFile);
    setVideoStatus(STATUS["LOADING"]);
    const res = await axios({
      method: 'post',
      url: 'http://127.0.0.1:5000/upload',
      data: formData,
      headers: {'Content-Type': 'multipart/form-data'}
    });
    // res.
    setVideoStatus(STATUS["DOWNLOADABLE"]);
    console.log(res);
    setResult(res.data);
    setSelectedFile(null);
  }

  //파일변환 완료에 대한 임시 버튼
  const onFileChanged = () => {
    setSelectedFile(null);
    setVideoStatus(STATUS["DOWNLOADABLE"]);
  }

  //변환된 파일 다운로드 코드
  const onFileDownload = () => {
    setSelectedFile(null);
  }

  return (
    <div className='container'>
      <h2>title</h2>
      <div>
        <br></br>
        <input type = "file" onChange = {onFileChange} />
      </div>
      <ActionButton 
        selectedFile={selectedFile}
        videoStatus={videoStatus}
        onFileUpload={onFileUpload}
        onFileChanged={onFileChanged}
        onFileDownload={onFileDownload}
      />
      <FileData 
        selectedFile={selectedFile}
        videoStatus={videoStatus} 
      />
      <br/>
    </div>
  )
}

export default App;